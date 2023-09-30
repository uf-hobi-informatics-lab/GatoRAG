import warnings

import torch
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer
from splade.models.transformer_rep import Splade
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BgeEncoder:
    PROMPT = "Represent this sentence for searching relevant passages:"

    def __init__(
        self,
        model_name="BAAI/bge-base-en",
    ):
        self.model = SentenceTransformer(model_name, device=DEVICE)

    def get_bge_embedding_single_sample(self, text4emb, is_qurey=False):
        if is_qurey:
            text4emb = " ".join([self.PROMPT, text4emb])
        return self.model.encode(text4emb, normalize_embeddings=False).tolist()


class InstructorEncoder:
    INSTRUCT_CONTEXT = "Represent the paragraph for retrieval: "
    INSTRUCT_QUERY = "Represent the question for retrieving supporting paragraphs: "

    def __init__(
        self,
        model_name="hkunlp/instructor-large",
        instruction_context=None,
        instruction_query=None,
    ) -> None:
        self.model = INSTRUCTOR(model_name)
        self.intruction_context = (
            instruction_context if instruction_context else self.INSTRUCT_CONTEXT
        )
        self.instruction_query = (
            instruction_query if instruction_query else self.INSTRUCT_QUERY
        )

    def get_instructor_embeddings_single_sample(self, query="", is_query=True):
        instruction = self.instruction_query if is_query else self.intruction_context
        model_inputs = [[instruction, query]]
        return self.model.encode(model_inputs, device=DEVICE, batch_size=1).tolist()[0]

    def get_instructor_embeddings_batch(self, queries=None, is_query=True):
        if queries is None:
            raise ValueError("query cannot be None")

        if not isinstance(queries, list):
            raise ValueError("queries must be list of strings")

        if DEVICE != "cuda":
            warnings.warn(
                "No CUDA available, processing batch will be slower than processing single sample."
            )

        instruction = self.instruction_query if is_query else self.intruction_context
        model_inputs = [[instruction, query] for query in queries]
        return self.model.encode(model_inputs, device=DEVICE).tolist()


class SpladeEncoder:
    def __init__(
        self, splade_name="naver/splade-cocondenser-ensembledistil", max_len=512
    ):
        self.model = Splade(splade_name, agg="max")
        self.tokenizer = AutoTokenizer.from_pretrained(splade_name)
        self.max_len = max_len
        self.model.eval()
        self.model.to(DEVICE)

    def _get_feature_embeddings(self, model_input, is_query):
        with torch.no_grad():
            if is_query:
                sparse_embeddings = self.model(d_kwargs=model_input)["d_rep"]
            else:
                sparse_embeddings = self.model(q_kwargs=model_input)["q_rep"]

        cols = [
            embedding.nonzero().squeeze().cpu().tolist()
            for embedding in sparse_embeddings
        ]
        weights = [
            embedding[cols[idx]].cpu().tolist()
            for idx, embedding in enumerate(sparse_embeddings)
        ]

        return cols, weights

    def get_splade_features_single_sample(self, query="", is_query=False):
        model_input = self.tokenizer(
            query,
            max_length=self.max_len,
            padding=False,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)

        cols, weights = self._get_feature_embeddings(model_input, is_query)

        return [
            {f"feature_{c}": w for c, w in zip(col, weight)}
            for col, weight in zip(cols, weights)
        ][0]

    def get_splade_features_batch(self, queries=None, is_query=False):
        if queries is None:
            raise ValueError("query cannot be None")

        if not isinstance(queries, list):
            raise ValueError("queries must be list of strings")

        if DEVICE != "cuda":
            warnings.warn(
                "No CUDA available, processing batch will be slower than processing single sample."
            )

        model_input = self.tokenizer(
            queries,
            max_length=self.max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)

        cols, weights = self._get_feature_embeddings(model_input, is_query)

        return [
            {f"feature_{c}": w for c, w in zip(col, weight)}
            for col, weight in zip(cols, weights)
        ]
