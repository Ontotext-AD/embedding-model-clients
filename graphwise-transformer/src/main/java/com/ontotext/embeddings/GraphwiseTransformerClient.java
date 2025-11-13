package com.ontotext.embeddings;

import ai.graphwise.transformer.GraphwiseTransformer;
import ai.graphwise.transformer.InferenceServiceGrpc;
import com.ontotext.Config;
import com.ontotext.embeddings.security.AuthClientInterceptor;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.exception.AuthenticationException;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;

import java.io.Closeable;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class GraphwiseTransformerClient implements EmbeddingModel, Closeable {

  public static final String MODEL_NAME_PROPERTY = "graphwise.transformer.embedding.model.name";
  public static final String MODEL_NAME_DEFAULT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2";
  public static final String ADDRESS_PROPERTY = "graphwise.transformer.address";
  public static final String ADDRESS_DEFAULT = "localhost:5050";
  public static final String BATCH_SIZE_PROPERTY = "graphwise.transformer.batch.size";
  public static final int BATCH_SIZE_DEFAULT = 256;
  public static final String AUTH_TOKEN_SECRET_PROPERTY = "graphwise.transformer.auth.token.secret";
  public static final String THREAD_POOL_SIZE_PROPERTY = "graphwise.transformer.thread.pool.size";

  private static final String MODEL_NAME = Config.getProperty(MODEL_NAME_PROPERTY,  MODEL_NAME_DEFAULT);
  private static final String ADDRESS = Config.getProperty(ADDRESS_PROPERTY, ADDRESS_DEFAULT);
  private static final int BATCH_SIZE = Config.getPropertyInt(BATCH_SIZE_PROPERTY, BATCH_SIZE_DEFAULT) * 1024;
  private static final int THREAD_POOL_SIZE = Config.getPropertyInt(THREAD_POOL_SIZE_PROPERTY, -1);

  private final ManagedChannel channel;
  private final InferenceServiceGrpc.InferenceServiceBlockingStub stub;
  private final ExecutorService executor;

  public GraphwiseTransformerClient() {
    this.channel = buildChannel();
    this.stub = InferenceServiceGrpc.newBlockingStub(channel);
    this.executor = createExecutor();
  }

  @Override public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
    List<List<TextSegment>> batches = chunkByBytes(segments);

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

    try {
      List<Embedding> allEmbeddings = futures.stream()
              .map(CompletableFuture::join)
              .flatMap(List::stream)
              .toList();
      return Response.from(allEmbeddings);
    } catch (CompletionException e) {
      Throwable cause = e.getCause();
      if (cause instanceof StatusRuntimeException sre
              && sre.getStatus().getCode() == Status.Code.UNAUTHENTICATED) {
        throw new AuthenticationException(
                "Authentication failed. Please verify shared secret.");
      }
      throw e;
    }
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

  private ManagedChannel buildChannel() {
    String[] parts = ADDRESS.split(":");
    String host = parts[0];
    int port = Integer.parseInt(parts[1]);
    String secret = Config.getProperty(AUTH_TOKEN_SECRET_PROPERTY);
    ManagedChannelBuilder<?> builder = ManagedChannelBuilder
            .forAddress(host, port)
            .usePlaintext();
    if (secret != null && !secret.isEmpty()) {
      builder = builder.intercept(new AuthClientInterceptor(secret));
    }
    return builder.build();
  }

  private ExecutorService createExecutor() {
    int threadPoolSize = THREAD_POOL_SIZE >= 0 ? THREAD_POOL_SIZE : Runtime.getRuntime().availableProcessors();
    return new ThreadPoolExecutor(
            Math.min(threadPoolSize, 2), threadPoolSize,
            30L, TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(threadPoolSize * 2),
            r -> {
              Thread t = new Thread(r);
              t.setDaemon(true);
              t.setName("graphwise-embedding-" + MODEL_NAME);
              return t;
            },
            new ThreadPoolExecutor.CallerRunsPolicy() // backpressure
    );
  }

  private List<List<TextSegment>> chunkByBytes(List<TextSegment> segments) {
    List<List<TextSegment>> chunks = new ArrayList<>();
    List<TextSegment> currentBatch = new ArrayList<>();
    int currentBytes = 0;

    for (TextSegment segment : segments) {
      int size = segment.text().getBytes(StandardCharsets.UTF_8).length
              + 32; // 32 bytes = message framing overhead

      // If adding this text would exceed our safe batch limit
      if (currentBytes + size > BATCH_SIZE) {
        if (!currentBatch.isEmpty()) {
          chunks.add(List.copyOf(currentBatch));
          currentBatch.clear();
          currentBytes = 0;
        }

        // If single item is huge (> limit), force it as its own batch
        if (size > BATCH_SIZE) {
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
}