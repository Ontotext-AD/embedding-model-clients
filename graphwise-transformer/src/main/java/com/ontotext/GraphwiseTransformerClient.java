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
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class GraphwiseTransformerClient implements EmbeddingModel, Closeable {

  public static final String MODEL_NAME_PROPERTY = "graphwise.transformer.embedding.model.name";
  public static final String ADDRESS_PROPERTY = "graphwise.transformer.address";
  public static final String BATCH_SIZE_PROPERTY = "graphwise.transformer.batch.size";
  private static final String MODEL_NAME = Config.getProperty(MODEL_NAME_PROPERTY);
  private static final String ADDRESS = Config.getProperty(ADDRESS_PROPERTY);

  private final InferenceServiceGrpc.InferenceServiceBlockingStub stub;
  private final ManagedChannel channel;
  private final ExecutorService executor;
  private final int batchSize;

  public GraphwiseTransformerClient() {
    String[] parts = ADDRESS.split(":");
    String host = parts[0];
    int port = Integer.parseInt(parts[1]);
    this.channel = ManagedChannelBuilder
            .forAddress(host, port)
            .usePlaintext()
            .build();

    this.stub = InferenceServiceGrpc.newBlockingStub(channel);
    this.executor = createExecutor();
    this.batchSize = Config.getPropertyInt(BATCH_SIZE_PROPERTY, 256) * 1024;
  }

  private ExecutorService createExecutor() {
    return new ThreadPoolExecutor(
            2, 4,
            30L, TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(8),
            r -> {
              Thread t = new Thread(r);
              t.setDaemon(true);
              t.setName("graphwise-embedding-" + MODEL_NAME);
              return t;
            },
            new ThreadPoolExecutor.CallerRunsPolicy() // backpressure
    );
  }

  @Override public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
    List<List<TextSegment>> batches = chunkByBytes(segments, batchSize);

    List<CompletableFuture<List<Embedding>>> futures = batches.stream()
            .map(batch -> CompletableFuture.supplyAsync(() -> {
              GraphwiseTransformer.SentenceRequest request = GraphwiseTransformer.SentenceRequest.newBuilder()
                      .setModelName(MODEL_NAME)
                      .addAllTexts(batch.stream().map(TextSegment::text).toList())
                      .build();

              GraphwiseTransformer.SentenceResponse response = stub.embedSentence(request);

              return toLangchainEmbeddings(response.getEmbeddingsList());
            }, executor))
            .toList();

    List<Embedding> allEmbeddings = futures.stream()
            .map(CompletableFuture::join)
            .flatMap(List::stream)
            .toList();

    return Response.from(allEmbeddings);
  }

  private List<List<TextSegment>> chunkByBytes(List<TextSegment> segments, int maxBytes) {
    List<List<TextSegment>> chunks = new ArrayList<>();
    List<TextSegment> currentBatch = new ArrayList<>();
    int currentBytes = 0;

    for (TextSegment segment : segments) {
      int size = segment.text().getBytes(StandardCharsets.UTF_8).length
              + 32; // 32 bytes = message framing overhead

      // If adding this text would exceed our safe batch limit
      if (currentBytes + size > maxBytes) {
        if (!currentBatch.isEmpty()) {
          chunks.add(List.copyOf(currentBatch));
          currentBatch.clear();
          currentBytes = 0;
        }

        // If single item is huge (> limit), force it as its own batch
        if (size > maxBytes) {
          chunks.add(List.of(segment));
          continue;
        }
      }

      currentBatch.add(segment);
      currentBytes += size;
    }

    if (!currentBatch.isEmpty()) {
      chunks.add(List.copyOf(currentBatch));
    }

    return chunks;
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
    } finally {
      executor.shutdownNow();
    }
  }
}