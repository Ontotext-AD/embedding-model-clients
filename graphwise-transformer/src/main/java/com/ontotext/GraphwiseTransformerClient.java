package com.ontotext;

import ai.graphwise.transformer.GraphwiseTransformer;
import ai.graphwise.transformer.InferenceServiceGrpc;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

import java.io.Closeable;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public class GraphwiseTransformerClient implements EmbeddingModel, Closeable {

  public static final String MODEL_NAME_PROPERTY = "graphwise.transformer.embedding.model.name";
  public static final String ADDRESS_PROPERTY = "graphwise.transformer.address";
  private static final String MODEL_NAME = Config.getProperty(MODEL_NAME_PROPERTY);
  private static final String ADDRESS = Config.getProperty(ADDRESS_PROPERTY);

  private final InferenceServiceGrpc.InferenceServiceBlockingStub stub;
  private final ManagedChannel channel;

  public GraphwiseTransformerClient() {
    String[] parts = ADDRESS.split(":");
    String host = parts[0];
    int port = Integer.parseInt(parts[1]);
    this.channel = ManagedChannelBuilder
            .forAddress(host, port)
            .usePlaintext()
            .build();

    this.stub = InferenceServiceGrpc.newBlockingStub(channel);
  }

  @Override public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
    List<String> texts = segments.stream().map(TextSegment::text).collect(Collectors.toList());
    GraphwiseTransformer.SentenceRequest request = GraphwiseTransformer.SentenceRequest.newBuilder()
            .setModelName(MODEL_NAME)
            .addAllTexts(texts)
            .build();

    GraphwiseTransformer.SentenceResponse response = stub.embedSentence(request);
    List<Embedding> embeddings = toLangchainEmbeddings(response.getEmbeddingsList());
    return Response.from(embeddings);
  }

  private List<Embedding> toLangchainEmbeddings(List<GraphwiseTransformer.Embedding> protoEmbeddings) {
    return protoEmbeddings.stream()
            .map(protoEmb -> {
              List<Float> list = protoEmb.getEmbeddingList();
              float[] vector = new float[list.size()];
              for (int i = 0; i < list.size(); i++) {
                vector[i] = list.get(i);
              }
              return new Embedding(vector);
            })
            .toList();
  }

  @Override
  public void close() {
    channel.shutdown();
    try {
      if (!channel.awaitTermination(5, TimeUnit.SECONDS)) {
        channel.shutdownNow();
      }
    } catch (InterruptedException e) {
      channel.shutdownNow();
      Thread.currentThread().interrupt();
    }
  }
}