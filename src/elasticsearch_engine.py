import logging
import time
import warnings
from typing import Dict, List, Tuple

import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

from embedding import BgeEncoder, InstructorEncoder, SpladeEncoder


class ElasticSearchEngine:
    def __init__(
        self,
        index_name: str,
        es_client: Elasticsearch = None,
        keys: Dict[str, str] = {"title": "title", "body": "txt"},
        batch_size: int = 128,
        sleep_for: int = 2,
        splade: str = "",
        sentence_transformer: str = "",
        instructor: str = "",
    ) -> None:
        self.results = {}
        self.batch_size = batch_size
        self.sleep_for = sleep_for
        self.text_key = keys["body"]
        self.title_key = keys["title"]
        self.splade = SpladeEncoder(splade_name=splade) if splade else SpladeEncoder()
        self.sentence_transformer = (
            SentenceTransformerEncoder(model_name=sentence_transformer)
            if sentence_transformer
            else SentenceTransformerEncoder()
        )
        self.instructor = (
            InstructorEncoder(model_name=instructor)
            if instructor
            else InstructorEncoder()
        )
        self.gte = GteEncoder()
        self.bge = BgeEncoder()

        # create elastic search engine
        self.check_index_name(index_name=index_name)
        self.index_name = index_name
        self.es = es_client
        if not self.es:
            warnings.warn(
                "elasticsearch client is not created, consider to call set_clinet method."
            )

    def set_client(self, elastic_connect_info: Dict[str, str]):
        self.es = Elasticsearch(**elastic_connect_info)
        if not self.es.ping():
            raise RuntimeError(
                "Connect to elasticsearch failed. Check elasticsearch connect information?"
            )

    def search(self, queries, top_k, *args, **kwargs):
        query_ids = list(queries.keys())
        for query_id in query_ids:
            query = queries[query_id]
            result = self.customized_search(
                text=query, top_hits=top_k + 1, qtype=kwargs.get("qtype")
            )
            scores = {}
            for corpus_id, score in result["hits"]:
                if corpus_id != query_id:  # query doesnt return in results
                    scores[corpus_id] = score
                self.results[query_id] = scores

        return self.results

    def index(self, corpus):
        # dictionary structure = {_id: {title_key: title, text_key: text}}
        dictionary = {
            idx: {
                self.title_key: corpus[idx].get("title", None),
                self.text_key: corpus[idx].get("text", None),
            }
            for idx in list(corpus.keys())
        }
        progress = tqdm.tqdm(total=len(dictionary))
        self.bulk_add_to_index(
            generate_actions=self.generate_actions(dictionary=dictionary, update=False),
            progress=progress,
        )
        self.wait_for_refresh(self.sleep_for)

    def initialization(self):
        self.delete_index()
        self.wait_for_refresh(self.sleep_for)
        self.create_index()

    def create_index(self):
        logging.info(
            "Creating fresh Elasticsearch-Index named - {}".format(self.index_name)
        )
        try:
            mapping = {
                "mappings": {
                    "properties": {
                        self.title_key: {"type": "text", "analyzer": "english"},
                        self.text_key: {"type": "text", "analyzer": "english"},
                        "sparse_context": {
                            "type": "rank_features",
                            "positive_score_impact": True,
                        },
                        "sb_emb": {
                            "type": "dense_vector",
                            "dims": 768,
                            "index": True,
                            "similarity": "cosine",
                            "index_options": {
                                "type": "hnsw",
                                "m": 16,
                                "ef_construction": 512,
                            },
                        },
                        "ins_emb": {
                            "type": "dense_vector",
                            "dims": 768,
                            "index": True,
                            "similarity": "cosine",
                            "index_options": {
                                "type": "hnsw",
                                "m": 16,
                                "ef_construction": 512,
                            },
                        },
                        "gte_emb": {
                            "type": "dense_vector",
                            "dims": 768,
                            "index": True,
                            "similarity": "cosine",
                            "index_options": {
                                "type": "hnsw",
                                "m": 16,
                                "ef_construction": 512,
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
                                "ef_construction": 512,
                            },
                        },
                    }
                }
            }
            self.es.indices.create(index=self.index_name, body=mapping, ignore=[400])
        except Exception as e:
            logging.error(
                "Unable to create Index in Elastic Search. Reason: {}".format(e)
            )

    def bulk_add_to_index(self, generate_actions, progress):
        """Bulk indexing to elastic search using generator actions

        Args:
            generate_actions (generator function): generator function must be provided
            progress (tqdm.tqdm): tqdm progress_bar
        """
        for ok, action in streaming_bulk(
            client=self.es, index=self.index_name, actions=generate_actions
        ):
            if not ok:
                print(
                    "Unable to index {}: {}".format(
                        action["index"]["_id"], action["index"]["error"]
                    )
                )
            progress.update()
        progress.close()

    def delete_index(self):
        logging.info(
            "Deleting previous Elasticsearch-Index named - {}".format(self.index_name)
        )
        try:
            self.es.indices.delete(
                index=self.index_name, ignore=[400, 404]
            )  # 404: IndexDoesntExistException
        except Exception as e:
            logging.error(
                "Unable to create Index in Elastic Search. Reason: {}".format(e)
            )

    def generate_actions(
        self, dictionary: Dict[str, Dict[str, str]], update: bool = False
    ):
        """Iterator function for efficient addition to Elasticsearch
        Ref: https://stackoverflow.com/questions/35182403/bulk-update-with-pythons-elasticsearch
        """
        for _id, value in dictionary.items():
            text = value[self.text_key]
            sbe = self.sentence_transformer.get_sentence_transformer_embeddings_single_sample(
                query=text
            )
            inse = self.instructor.get_instructor_embeddings_single_sample(
                query=text, is_query=False
            )
            spe = self.splade.get_splade_features_single_sample(query=text)
            gte_emb = self.gte.get_sentence_transformer_embeddings_single_sample(text)
            bge_emb = self.bge.get_bge_embedding_single_sample(text, is_qurey=False)

            if not update:
                doc = {
                    "_id": str(_id),
                    "_op_type": "index",
                    "refresh": "wait_for",
                    self.text_key: value[self.text_key],
                    self.title_key: value[self.title_key],
                    "sparse_context": spe,
                    "sb_emb": sbe,
                    "ins_emb": inse,
                    "gte_emb": gte_emb,
                    "bge_emb": bge_emb,
                }
            else:
                doc = {
                    "_id": str(_id),
                    "_op_type": "update",
                    "refresh": "wait_for",
                    "doc": {
                        self.text_key: value[self.text_key],
                        self.title_key: value[self.title_key],
                        "sparse_context": spe,
                        "sb_emb": sbe,
                        "ins_emb": inse,
                        "gte_emb": gte_emb,
                        "bge_emb": bge_emb,
                    },
                }
            yield doc

    def check_index_name(self, index_name="default_name"):
        """Check Elasticsearch Index Name"""
        # https://stackoverflow.com/questions/41585392/what-are-the-rules-for-index-names-in-elastic-search
        # Check 1: Must not contain the characters ===> #:\/*?"<>|,
        for char in '#:\/*?"<>|,':
            if char in index_name:
                raise ValueError(
                    'Invalid Elasticsearch Index, must not contain the characters ===> #:\/*?"<>|,'
                )

        # Check 2: Must not start with characters ===> _-+
        if index_name.startswith(("_", "-", "+")):
            raise ValueError(
                "Invalid Elasticsearch Index, must not start with characters ===> _ or - or +"
            )

        # Check 3: must not be . or ..
        if index_name in [".", ".."]:
            raise ValueError("Invalid Elasticsearch Index, must not be . or ..")

        # Check 4: must be lowercase
        if not index_name.islower():
            raise ValueError("Invalid Elasticsearch Index, must be lowercase")

    def hit_template(
        self, es_res: Dict[str, object], hits: List[Tuple[str, float]]
    ) -> Dict[str, object]:
        """Hit output results template

        Args:
            es_res (Dict[str, object]): Elasticsearch response
            hits (List[Tuple[str, float]]): Hits from Elasticsearch

        Returns:
            Dict[str, object]: Hit results
        """
        result = {
            "meta": {
                "total": es_res["hits"]["total"]["value"],
                "took": es_res["took"],
                "num_hits": len(hits),
            },
            "hits": hits,
        }
        return result

    def customized_search(
        self, text: str, top_hits: int, skip: int = 0, qtype: int = 0
    ) -> Dict[str, object]:
        if qtype > 1:
            st_emb = self.sentence_transformer.get_sentence_transformer_embeddings_single_sample(
                query=text
            )
            ins_emb = self.instructor.get_instructor_embeddings_single_sample(
                query=text, is_query=True
            )
            gte_emb = self.gte.get_sentence_transformer_embeddings_single_sample(text)
            bge_emb = self.bge.get_bge_embedding_single_sample(text, is_qurey=True)
            sparse_query = [
                {
                    "rank_feature": {
                        "field": f"sparse_context.{k}",
                        "linear": {},
                        "boost": v,
                    }
                }
                for k, v in self.splade.get_splade_features_single_sample(
                    query=text
                ).items()
            ]
            sparse_query_bm25 = [
                {"match": {self.text_key: text}},
                {"match": {self.title_key: text}},
            ] + sparse_query
        else:
            st_emb = []
            ins_emb = []
            sparse_query = []
            sparse_query_bm25 = []

        if qtype == 1:
            req_body = {
                "query": {
                    "bool": {
                        "should": [
                            {"match": {self.text_key: text}},
                            {"match": {self.title_key: text}},
                        ]
                    }
                }
            }
        elif qtype == 2:
            req_body = {
                "knn": {
                    "field": "sb_emb",
                    "query_vector": st_emb,
                    "k": 20,
                    "num_candidates": 100,
                },
            }
        elif qtype == 3:
            req_body = {
                "knn": {
                    "field": "ins_emb",
                    "query_vector": ins_emb,
                    "k": 20,
                    "num_candidates": 100,
                },
            }
        elif qtype == 4:
            req_body = {
                "query": {
                    "bool": {
                        "should": sparse_query,
                        "boost": 0.5,
                        "minimum_should_match": 1,
                    }
                }
            }
        elif qtype == 5:
            req_body = {
                "knn": [
                    {
                        "field": "sb_emb",
                        "query_vector": st_emb,
                        "k": 20,
                        "num_candidates": 100,
                        "boost": 0.5,
                    },
                    {
                        "field": "ins_emb",
                        "query_vector": ins_emb,
                        "k": 10,
                        "num_candidates": 100,
                        "boost": 0.5,
                    },
                ],
            }
        elif qtype == 6:
            req_body = {
                "query": {"bool": {"should": sparse_query_bm25}},
                "knn": {
                    "field": "ins_emb",
                    "query_vector": ins_emb,
                    "k": 10,
                    "num_candidates": 100,
                },
            }
        elif qtype == 7:
            req_body = {
                "query": {
                    "bool": {
                        "should": sparse_query_bm25,
                        "boost": 0.2,
                        "minimum_should_match": 1,
                    },
                },
                "knn": [
                    {
                        "field": "sb_emb",
                        "query_vector": st_emb,
                        "k": 10,
                        "num_candidates": 100,
                        "boost": 0.3,
                    },
                    {
                        "field": "ins_emb",
                        "query_vector": ins_emb,
                        "k": 10,
                        "num_candidates": 100,
                        "boost": 0.5,
                    },
                ],
            }
        elif qtype == 8:
            req_body = {
                "query": {
                    "bool": {
                        "should": sparse_query,
                        "boost": 0.2,
                        "minimum_should_match": 1,
                    },
                },
                "knn": [
                    {
                        "field": "sb_emb",
                        "query_vector": st_emb,
                        "k": 10,
                        "num_candidates": 100,
                        "boost": 0.3,
                    },
                    {
                        "field": "ins_emb",
                        "query_vector": ins_emb,
                        "k": 10,
                        "num_candidates": 100,
                        "boost": 0.5,
                    },
                ],
            }
        elif qtype == 9:
            req_body = {
                "knn": {
                    "field": "gte_emb",
                    "query_vector": gte_emb,
                    "k": 20,
                    "num_candidates": 100,
                },
            }
        elif qtype == 10:
            req_body = {
                "knn": {
                    "field": "bge_emb",
                    "query_vector": bge_emb,
                    "k": 20,
                    "num_candidates": 100,
                },
            }
        elif qtype == 11:
            req_body = {
                "query": {
                    "bool": {
                        "should": sparse_query,
                        "minimum_should_match": 1,
                    },
                },
                "knn": {
                    "field": "gte_emb",
                    "query_vector": gte_emb,
                    "k": 20,
                    "num_candidates": 100,
                },
            }
        elif qtype == 12:
            req_body = {
                "query": {
                    "bool": {
                        "should": [
                            {"match": {self.text_key: text}},
                            {"match": {self.title_key: text}},
                        ],
                        "minimum_should_match": 1,
                    },
                },
                "knn": {
                    "field": "bge_emb",
                    "query_vector": bge_emb,
                    "k": 20,
                    "num_candidates": 100,
                },
            }
        elif qtype == 13:
            req_body = {
                "query": {
                    "bool": {
                        "should": [
                            {"match": {self.text_key: text}},
                            {"match": {self.title_key: text}},
                        ]
                    }
                },
                "knn": {
                    "field": "ins_emb",
                    "query_vector": ins_emb,
                    "k": 20,
                    "num_candidates": 100,
                },
            }
        else:
            req_body = {
                "query": {
                    "multi_match": {
                        "query": text,
                        "type": "best_fields",
                        "fields": [self.text_key, self.title_key],
                        "tie_breaker": 0.5,
                    }
                }
            }

        res = self.es.search(
            index=self.index_name,
            body=req_body,
            size=skip + top_hits,
            request_timeout=360,
        )

        hits = []

        for hit in res["hits"]["hits"][skip:]:
            hits.append((hit["_id"], hit["_score"]))

        return self.hit_template(es_res=res, hits=hits)

    @staticmethod
    def wait_for_refresh(seconds):
        if seconds:
            time.sleep(seconds)
