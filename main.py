import pandas as pd
import numpy as np
import logging
import os
import asyncio
import warnings
from typing import Dict, Any
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import hdbscan

# Фильтрация предупреждений
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clustering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Инициализация окружения и проверка CUDA"""
    logger.info(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        logger.info(f"CUDA available: True")
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        torch.backends.cudnn.benchmark = True
    else:
        logger.info("CUDA available: False (работаем на CPU)")


async def load_excel(file_path: str) -> pd.DataFrame:
    """Загрузка и подготовка данных"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден!")

        df = pd.read_excel(file_path, engine='openpyxl')
        logger.info(f"Загружено строк: {len(df)}")

        initial_count = len(df)
        df = df.drop_duplicates(subset=['keyword'], keep='first')
        logger.info(f"Удалено дубликатов: {initial_count - len(df)}")

        return df
    except Exception as e:
        logger.error(f"Ошибка загрузки: {str(e)}")
        raise


async def get_bert_embeddings(texts: list) -> np.ndarray:
    """Генерация эмбеддингов с использованием GPU"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Используемое устройство: {device}")

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)
        model.eval()

        embeddings = []
        batch_size = 256 if torch.cuda.is_available() else 128

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Обработка батча {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = model(**encoded)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)

            await asyncio.sleep(0.001)

        return np.vstack(embeddings)
    except Exception as e:
        logger.error(f"Ошибка генерации эмбеддингов: {str(e)}")
        raise


def cluster_data(embeddings: np.ndarray, method: str, params: dict) -> np.ndarray:
    """Оптимизированная кластеризация"""
    try:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(embeddings)

        if method == 'kmeans':
            model = KMeans(
                n_clusters=params['n_clusters'],
                random_state=42,
                n_init='auto'
            )
        elif method == 'hdbscan':
            reducer = umap.UMAP(
                n_components=params['umap_components'],
                random_state=42,
                n_jobs=-1
            )
            reduced = reducer.fit_transform(scaled)

            model = hdbscan.HDBSCAN(
                min_cluster_size=params['min_cluster_size'],
                cluster_selection_epsilon=params['epsilon'],
                min_samples=params['min_samples'],
                core_dist_n_jobs=-1
            )
            scaled = reduced

        return model.fit_predict(scaled)
    except Exception as e:
        logger.error(f"Ошибка кластеризации: {str(e)}")
        raise


async def main(config: dict):
    """Основной процесс выполнения"""
    try:
        setup_environment()

        logger.info("Начало загрузки данных...")
        df = await load_excel(config['input'])
        texts = df['keyword'].tolist()

        logger.info("Генерация BERT-эмбеддингов...")
        embeddings = await get_bert_embeddings(texts)
        logger.info(f"Размерность эмбеддингов: {embeddings.shape}")

        logger.info("Запуск кластеризации...")
        clusters = cluster_data(embeddings, config['method'], config)
        df['cluster'] = clusters

        output_file = 'results.xlsx'
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Данные', index=False)

            summary = df.groupby('cluster').agg(
                Ключевые_слова=('keyword', 'count'),
                Топ_ключей=('keyword', lambda x: ', '.join(map(str, x.head(5)))),
                Общая_частотность=('frequency', 'sum')
            ).reset_index()

            summary.to_excel(writer, sheet_name='Статистика', index=False)

        logger.info(f"Результаты сохранены в {output_file}")
        logger.info("Процесс успешно завершен!")

    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Конфигурация
config = {
    'input': 'keywords.xlsx',
    'method': 'hdbscan',
    'min_cluster_size': 5,
    'epsilon': 0.3,
    'min_samples': 5,
    'n_clusters': 25,
    'umap_components': 5
}

if __name__ == "__main__":
    try:
        asyncio.run(main(config))
    except Exception as e:
        logger.error(f"Ошибка запуска: {str(e)}")
