import logging
import time
import warnings
from typing import Dict, List, Tuple

import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

from .embedding import BgeEncoder, InstructorEncoder, SpladeEncoder


class ElasticSearchEngine:
    def __init__(
        self,
        index_name: str,
        es_client: Elasticsearch = None,
        sleep_for: int = 2,
        splade: str = "",
        instructor: str = "",
        bge: str = "",
        use_rff: bool = False,
        rff_k: int = 61,
    ):
        self.sleep_for = sleep_for
        self.splade = SpladeEncoder(splade_name=splade) if splade else SpladeEncoder()
        self.instructor = InstructorEncoder(model_name=instructor) if instructor else InstructorEncoder()
        self.bge = BgeEncoder(model_name=bge) if bge else BgeEncoder()

        # create elastic search engine
        self.check_index_name(index_name=index_name)
        self.index_name = index_name
        self.es = es_client
        if not self.es:
            warnings.warn(
                "Elasticsearch client is None." "Use `ElasticSearchEngine.set_clinet` method to setup client"
            )
        self.mapping = {
            "mappings": {
                "properties": {
                    "text": {"type": "text", "analyzer": "english"},
                    "sparse_context": {
                        "type": "rank_features",
                        "positive_score_impact": True,
                    },
                    "instruct_emb": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": True,
                        "similarity": "cosine",
                        "index_options": {
                            "type": "hnsw",
                            "m": 8,
                            "ef_construction": 128,
                        },
                    },
                    "bge_emb": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": True,
                        "similarity": "cosine",
                        "index_options": {
                            "type": "hnsw",
                            "m": 8,
                            "ef_construction": 128,
                        },
                    },
                }
            }
        }

    def set_client(self, elastic_connect_info: Dict[str, str]):
        self.es = Elasticsearch(**elastic_connect_info)
        if not self.es.ping():
            raise RuntimeError("Connect to elasticsearch failed. Check elasticsearch connect information?")

    def initialization(self):
        if self.es.indices.exists(index=self.index_name):
            self.delete_index()
        self.wait_for_refresh(self.sleep_for)

    def create_index(self, customized_mapping=None):
        logging.info("Creating fresh Elasticsearch-Index named - {}".format(self.index_name))
        try:
            if customized_mapping:
                self.mapping = customized_mapping
            self.es.indices.create(index=self.index_name, body=self.mapping, ignore=[400])
        except Exception as e:
            logging.error("Unable to create Index in Elastic Search. Reason: {}".format(e))

    def index(self, corpus):
        progress = tqdm.tqdm(total=len(corpus))
        self.bulk_add_to_index(
            generate_actions=self.generate_actions(samples=corpus, update=False),
            progress=progress,
        )
        self.wait_for_refresh(self.sleep_for)

    def update(self, corpus):
        progress = tqdm.tqdm(total=len(corpus))
        self.bulk_add_to_index(
            generate_actions=self.generate_actions(samples=corpus, update=True),
            progress=progress,
        )
        self.wait_for_refresh(self.sleep_for)

    def delete_index(self):
        flag = input(f"Are you sure you want to delete the index: {self.index_name} - y/n")
        if flag.lower() == "n":
            return
        logging.info("Deleting previous Elasticsearch-Index named - {}".format(self.index_name))
        try:
            self.es.indices.delete(index=self.index_name, ignore=[400, 404])  # 404: IndexDoesntExistException
        except Exception as e:
            logging.error("Unable to create Index in Elastic Search. Reason: {}".format(e))

    def bulk_add_to_index(self, generate_actions, progress):
        """Bulk indexing to elastic search using generator actions

        Args:
            generate_actions (generator function): generator function must be provided
            progress (tqdm.tqdm): tqdm progress_bar
        """
        for ok, action in streaming_bulk(client=self.es, index=self.index_name, actions=generate_actions):
            if not ok:
                print("Unable to index {}: {}".format(action["index"]["_id"], action["index"]["error"]))
            progress.update()
        progress.close()

    def generate_actions(self, samples, update=False):
        """Iterator function for efficient addition to Elasticsearch
        Ref: https://stackoverflow.com/questions/35182403/bulk-update-with-pythons-elasticsearch
        """
        for sample in samples:
            text = sample["text"]
            inse = self.instructor.get_instructor_embeddings_single_sample(query=text, is_query=False)
            spe = self.splade.get_splade_features_single_sample(query=text)
            bge_emb = self.bge.get_bge_embedding_single_sample(text, is_qurey=False)

            if update:
                _id = sample["_id"]
                doc = {
                    "_id": str(_id),
                    "_op_type": "update",
                    "refresh": "wait_for",
                    "doc": {
                        **{k: v for k, v in sample.items() if k != "_id"},
                        "sparse_context": spe,
                        "instruct_emb": inse,
                        "bge_emb": bge_emb,
                    },
                }
            else:
                doc = {
                    **sample,
                    "_op_type": "index",
                    "refresh": "wait_for",
                    "sparse_context": spe,
                    "instruct_emb": inse,
                    "bge_emb": bge_emb,
                }

            yield doc

    def check_index_name(self, index_name="default_name"):
        """Check Elasticsearch Index Name"""
        # https://stackoverflow.com/questions/41585392/what-are-the-rules-for-index-names-in-elastic-search
        # Check 1: Must not contain the characters ===> #:\/*?"<>|,
        for char in '#:\/*?"<>|,':
            if char in index_name:
                raise ValueError('Invalid Elasticsearch Index, must not contain the characters ===> #:\/*?"<>|,')

        # Check 2: Must not start with characters ===> _-+
        if index_name.startswith(("_", "-", "+")):
            raise ValueError("Invalid Elasticsearch Index, must not start with characters ===> _ or - or +")

        # Check 3: must not be . or ..
        if index_name in [".", ".."]:
            raise ValueError("Invalid Elasticsearch Index, must not be . or ..")

        # Check 4: must be lowercase
        if not index_name.islower():
            raise ValueError("Invalid Elasticsearch Index, must be lowercase")

    def hit_template(self, es_res: Dict[str, object], hits: List[Tuple[str, float]]) -> Dict[str, object]:
        """Hit output results template

        Args:
            es_res (Dict[str, object]): Elasticsearch response
            hits (List[Tuple[str, float]]): Hits from Elasticsearch

        Returns:
            Dict[str, object]: Hit results
        """
        if es_res:
            result = {
                "meta": {
                    "total": es_res["hits"]["total"]["value"],
                    "took": es_res["took"],
                    "num_hits": len(hits),
                },
                "hits": hits,
            }
        else:
            result = {
                "meta": {
                    "total": 0,
                    "num_hits": len(hits),
                },
                "hits": hits,
            }
        return result

    def get_current_index_mapping(self):
        return self.es.indices.get_mapping(index=self.index_name)[self.index_name]["mappings"]

    def get_query_embedding(self, query, embedding_method="bge"):
        if embedding_method == "bge":
            return self.bge.get_bge_embedding_single_sample(query, is_qurey=True)
        elif embedding_method == "instructor":
            return self.instructor.get_instructor_embeddings_single_sample(query=query, is_query=True)
        elif embedding_method == "sparse":
            return self.splade.get_splade_features_single_sample(query=query)
        else:
            raise NotImplementedError(f"Embedding method {embedding_method} is not implemented")

    def search(self, query, top_k, **kwargs):
        # add return field filter: we do not want to return any embeddings
        field_keys = self.get_current_index_mapping()["properties"].keys()
        fields = {k for k in field_keys if "emb" not in k}
        fields = [k for k in fields if k not in {"sparse_context", "refresh"}]

        query = {**query, "_source": fields}

        res = self.es.search(
            index=self.index_name,
            body=query,
            size=top_k,
            request_timeout=360,
        )

        hits = []
        for hit in res["hits"]["hits"]:
            hits.append(hit)

        return self.hit_template(es_res=res, hits=hits)

    @staticmethod
    def wait_for_refresh(seconds):
        if seconds:
            time.sleep(seconds)

    def _rrf(self):
        """implementation of Reciprocal Rank Fusion"""
        pass
