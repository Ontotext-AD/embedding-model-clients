# embedding-model-clients

This project provides a collection of clients for interacting with various embedding models. It includes modules for specific embedding model providers like [GraphWise Transformer](https://gitlab.ontotext.com/graphdb-team/graphwise-transformer) and OpenAI API.
Clients are implementations of langchain4j [EmbeddingModel](https://github.com/langchain4j/langchain4j/blob/main/langchain4j-core/src/main/java/dev/langchain4j/model/embedding/EmbeddingModel.java) interface. 
They can be used in systems that provide similarity search by creating embeddings from texts, 
such as GraphDB [Elasticsearch](https://graphdb.ontotext.com/documentation/11.2/elasticsearch-graphdb-connector.html) and [Opensearch](https://graphdb.ontotext.com/documentation/11.2/opensearch-graphdb-connector.html) Connectors.
Existing clients serve as examples and default implementations that GraphDB connectors use to provide similarity searches.
You can provide additional clients by implementing [EmbeddingModel](https://github.com/langchain4j/langchain4j/blob/main/langchain4j-core/src/main/java/dev/langchain4j/model/embedding/EmbeddingModel.java) interface 
and add them to your GraphDB distribution to use for similarity searches in GraphDB Connectors.

## Modules

* **assembly**: Create an assembly jar that contains clients and their dependencies.
* **embedding-clients-common**: Common code used by the other client modules.
* **graphwise-transformer-client**: A client for interacting with a [GraphWise Transformer](https://gitlab.ontotext.com/graphdb-team/graphwise-transformer).
* **openai-embedding-client**: A client for interacting with the OpenAI embedding API.

## Building

To build the project, you can use Maven. Run the following command from the project's root directory:

```bash
mvn clean install
```

## Installation

1.  Build the project to create the assembly JAR.
2.  Copy the generated JAR from `./assembly/target/embedding-model-clients-assembly-{project.version}.jar` to your application's classpath.
  *   For GraphDB Connectors, place the JAR in the directory of the respective connector, for example: `dist/graphdb/target/graphdb/lib/plugins/elasticsearch-connector/`.

## Configuration

Clients are configured via system properties.

### GraphwiseTransformerClient

| Property                                     | Description                                               | Default                                                       |
| -------------------------------------------- |-----------------------------------------------------------| ------------------------------------------------------------- |
| `graphwise.transformer.address`              | The host and port of the GraphWise Transformer service.   | `localhost:5050`                                              |
| `graphwise.transformer.embedding.model.name` | The name of the sentence transformer model to use.        | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `graphwise.transformer.batch.size`           | The maximum request batch size in kilobytes.              | `256`                                                         |
| `graphwise.transformer.auth.token.secret`    | Shared secret for authentication.                         | `null`                                                        |
| `graphwise.transformer.thread.pool.size`     | The size of the client-side thread pool.                  | Number of available processors                                |

### OpenAIEmbeddingClient

| Property            | Description              | Default                  |
|---------------------|--------------------------|--------------------------|
| `openai.api.key`    | Your OpenAI API key.     | `null`                   |
| `openai.model.name` | The OpenAI model to use. | `text-embedding-ada-002` |

## Usage

Once the JAR is on the classpath and the necessary properties are configured, you can use the clients in GraphDB Connectors by specifying the fully qualified class name of the desired implementation in the connector configuration.

**Example values for `embeddingModel` parameter:**
*   `com.ontotext.embedding.GraphwiseTransformerClient`
*   `com.ontotext.embedding.OpenAIEmbeddingClient`

## License

Licensed under Apache 2.0.

