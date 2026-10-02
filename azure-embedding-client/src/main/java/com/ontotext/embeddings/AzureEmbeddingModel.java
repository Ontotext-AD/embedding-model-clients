package com.ontotext.embeddings;

import com.azure.identity.ClientSecretCredentialBuilder;
import com.azure.identity.DefaultAzureCredentialBuilder;
import com.ontotext.graphdb.Config;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.azure.AzureOpenAiEmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class AzureEmbeddingModel implements EmbeddingModel {

    public static final String MODEL_NAME_PROPERTY = "azure.embedding.model.name";
    public static final String API_KEY_PROPERTY = "azure.embedding.model.api.key";
    public static final String TENANT_ID_PROPERTY = "azure.embedding.tenant.id";
    public static final String CLIENT_ID_PROPERTY = "azure.embedding.client.id";
    public static final String CLIENT_SECRET_PROPERTY = "azure.embedding.client.secret";
    public static final String DIMENSIONS_PROPERTY = "azure.embedding.model.dimensions";
    public static final String URI_PROPERTY = "azure.embedding.model.baseUrl";
    private static final Logger LOGGER = LoggerFactory.getLogger(AzureEmbeddingModel.class);

    private final EmbeddingModel embeddingModel;
    private final String modelName;

    public AzureEmbeddingModel(String modelName) {
        this.modelName = resolveModelName(modelName);
        this.embeddingModel = createEmbeddingModel();
    }

    //Used in GraphDB 11.2.x and 11.3.x
    @Deprecated
    public AzureEmbeddingModel() {
        this(Config.getProperty(MODEL_NAME_PROPERTY));
    }

    private String resolveModelName(String modelName) {
        if (modelName == null || modelName.isEmpty()) {
            modelName = Config.getProperty(MODEL_NAME_PROPERTY);
        }
        LOGGER.info("Creating instance using model: {}", modelName);
        return modelName;
    }

    private EmbeddingModel createEmbeddingModel() {
        String uri = null; // Let it throw the Azure error if null.
        if (Config.getProperty(URI_PROPERTY) != null) {
            // Strip trailing slash.
            uri = Config.getProperty(URI_PROPERTY).replaceAll("/+$", "");
        }
        AzureOpenAiEmbeddingModel.Builder builder =
            AzureOpenAiEmbeddingModel.builder().deploymentName(modelName).endpoint(uri);

        String apiKey = Config.getProperty(API_KEY_PROPERTY);
        String tenantId = Config.getProperty(TENANT_ID_PROPERTY);
        String clientId = Config.getProperty(CLIENT_ID_PROPERTY);
        String clientSecret = Config.getProperty(CLIENT_SECRET_PROPERTY);

        if (apiKey != null && !apiKey.trim().isEmpty()) {
            LOGGER.info("Using API Key authentication for Azure OpenAI.");
            builder.apiKey(apiKey);
        } else if (tenantId != null && clientId != null && clientSecret != null) {
            LOGGER.info("Using properties `azure.embedding.tenant.id`, `azure.embedding.client.id`"
                + "and `azure.embedding.client.secret` for authentication for Azure OpenAI.");
            builder.tokenCredential(
                new ClientSecretCredentialBuilder()
                    .tenantId(tenantId)
                    .clientId(clientId)
                    .clientSecret(clientSecret)
                    .build()
            );
        } else {
            LOGGER.info("Can't configure Azure OpenAI with the provided properties. "
                + "Falling back to DefaultAzureCredential.");
            builder.tokenCredential(new DefaultAzureCredentialBuilder().build());
        }

        String dimensions = Config.getProperty(DIMENSIONS_PROPERTY);
        if (dimensions != null && !dimensions.isEmpty()) {
            builder.dimensions(Integer.parseInt(dimensions));
        }
        LOGGER.info("Creating Azure Embedding model with endpoint {} and deployment {}.", uri,
                modelName);
        if (LOGGER.isDebugEnabled()) {
            LOGGER.info("Will log requests and responses.");
            builder.logRequestsAndResponses(true);
        }
        return builder.build();
    }

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> list) {
        return embeddingModel.embedAll(list);
    }
}
