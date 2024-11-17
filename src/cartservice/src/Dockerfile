# Build stage
FROM mcr.microsoft.com/dotnet/sdk:8.0.203 AS builder
WORKDIR /app

# Copy project file and restore dependencies
COPY cartservice.csproj .
RUN dotnet restore cartservice.csproj -r linux-musl-arm64

# Copy source code and publish
COPY . .
RUN dotnet publish cartservice.csproj \
    -p:PublishSingleFile=true \
    -r linux-musl-arm64 \
    --self-contained true \
    -p:PublishTrimmed=True \
    -p:TrimMode=Full \
    -c release \
    -o /cartservice \
    --no-restore

# Runtime stage
FROM mcr.microsoft.com/dotnet/runtime-deps:8.0.3-alpine3.18
WORKDIR /app

# Copy published files from the builder stage
COPY --from=builder /cartservice .

# Set environment variables and expose ports
EXPOSE 7070
ENV DOTNET_EnableDiagnostics=0 \
    ASPNETCORE_HTTP_PORTS=7070

# Use a non-root user for better security
USER 1000

# Run the application
ENTRYPOINT ["/app/cartservice"]
