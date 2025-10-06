from elasticsearch import Elasticsearch , helpers
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime, timedelta

class ElasticFetcher:
    load_dotenv()
    def __init__(self, host=os.getenv("ES_HOST"), index=os.getenv("ES_INDEX"), username=os.getenv("ES_USERNAME"), password=os.getenv("ES_PASSWORD")):
        self.es = Elasticsearch(
            hosts=[host],
            basic_auth=(username, password) if username and password else None
        )
        self.index = index

    def fetch_all(self, size=10_000):
        """
        Récupère toutes les valeurs présentes pour tous les `names` dans l'index.
        """
        query = {
            "size": size,
            "query": {"match_all": {}}
        }

        resp = self.es.search(index=self.index, body=query, scroll='4m')
        hits = resp['hits']['hits']
        data = [hit["_source"] for hit in hits]

        # Pagination avec scroll si nécessaire
        scroll_id = resp.get('_scroll_id')
        while scroll_id and hits:
            resp = self.es.scroll(scroll_id=scroll_id, scroll='4m')
            hits = resp['hits']['hits']
            data.extend(hit['_source'] for hit in hits)

        df = pd.DataFrame(data)
        return df

   

    def fetch_last_days(self, size=10_000,day=12):
        now = datetime.utcnow()
        n_day_ago = now - timedelta(days=day)

        date_format = "%Y-%m-%d %H:%M:%S.%f"
        now_str = now.strftime(date_format)[:-3]  
        n_day_ago_str = n_day_ago.strftime(date_format)[:-3]

        query = {
            "size": size,
            "query": {
                "range": {
                    "date_mesure.keyword": {
                        "gte": n_day_ago_str,
                        "lte": now_str
                    }
                }
            },
            # attention ici aussi => on trie sur keyword
            "sort": [
                {"date_mesure.keyword": "asc"}
            ]
        }

        resp = self.es.search(index=self.index, body=query, scroll='4m')
        hits = resp['hits']['hits']
        data = [hit["_source"] for hit in hits]

        scroll_id = resp.get('_scroll_id')
        while scroll_id and hits:
            resp = self.es.scroll(scroll_id=scroll_id, scroll='4m')
            hits = resp['hits']['hits']
            data.extend(hit['_source'] for hit in hits)

        df = pd.DataFrame(data)
        return df


    def fetch_pred(self, size=10_000,day=12):
        now = datetime.utcnow()
        n_day_ago = now - timedelta(days=day)
        future_day = now + timedelta(days=5)


        date_format = "%Y-%m-%d %H:%M:%S.%f"
        now_str = now.strftime(date_format)[:-3]  
        n_day_ago_str = n_day_ago.strftime(date_format)[:-3]
        future_day_str = future_day.strftime(date_format)[:-3]

        query = {
            "size": size,
            "query": {
                "range": {
                    "date_mesure.keyword": {
                        "gte": n_day_ago_str,
                        "lte": future_day_str
                    }
                }
            },
            # attention ici aussi => on trie sur keyword
            "sort": [
                {"date_mesure.keyword": "asc"}
            ]
        }

        resp = self.es.search(index=self.index, body=query, scroll='4m')
        hits = resp['hits']['hits']
        data = [hit["_source"] for hit in hits]

        scroll_id = resp.get('_scroll_id')
        while scroll_id and hits:
            resp = self.es.scroll(scroll_id=scroll_id, scroll='4m')
            hits = resp['hits']['hits']
            data.extend(hit['_source'] for hit in hits)

        df = pd.DataFrame(data)
        return df

    def save_to_es(self, df):
        """
        Envoie un DataFrame dans un index Elasticsearch.
        """
        
        index_name = os.getenv("ES_INDEX_dest")

        if df.empty:
            print(f"⚠️ Rien à indexer pour {index_name}")
            return

        actions = [
            {
                "_index": index_name,
                "_source": row.dropna().to_dict()
            }
            for _, row in df.iterrows()
        ]
        helpers.bulk(self.es, actions)
        print(f"✅ {len(actions)} documents saves ")
      
