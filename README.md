# GatorRAG
Use GatortronGPT for Retrieval Augmented Generation

## Embedding models
### Out-of-box models
- check MTEB leader board for the latest rank https://huggingface.co/spaces/mteb/leaderboard
- models for testing
    - sionic-ai/sionic-ai-v2
    - BAAI/bge-base-en-v1.5
### Gatortron based embedding model
- TODO: train gatortron BERT to perform embedding task


## ElastiSearch
### Set up elasticsearch python package
- current elasticsearch-py release has several features not supported (8.8 release)
- To run most advanced feature we recommand install elasticsearch-py from github main branch as `pip install git+https://github.com/elastic/elasticsearch-py.git`
### Set up Elasticsearch locally (docker based installation)
> https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html 
#### install elasticsearch on local machine
- install docker on your local (check docker website)
- get elasticsearch images `docker pull docker.elastic.co/elasticsearch/elasticsearch:8.8.2`
- create a new elasticsearch single node network `docker network create elastic`
- then we can create a tmux env to run docker elasticsearch service
- inside tmux `docker run --name es-node01 --net elastic -p 9200:9200 -p 9300:9300 -t docker.elastic.co/elasticsearch/elasticsearch:8.8.2`
- note: you need to wait until the service is up where there will be a password generated, you have to save it in a separate file which we need to use later for access via python client API
#### optional to set up kibana (which is a web client)
- use a separate tmux env
- `docker pull docker.elastic.co/kibana/kibana:8.8.2`
- `docker run --name kib-01 --net elastic -p 5601:5601 docker.elastic.co/kibana/kibana:8.8.2`
### Index for hybrid search
### qurey using hybrid search
