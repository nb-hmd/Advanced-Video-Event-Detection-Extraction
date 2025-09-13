# Automatic Event Detection and Extraction from Video - Technical Architecture Document

## 1. Architecture Design

```mermaid
graph TD
    A[User Interface] --> B[Video Processing Pipeline]
    B --> C[Phase 1: OpenCLIP Engine]
    B --> D[Phase 2: BLIP Re-ranker]
    B --> E[Phase 3: UniVTG Refiner]
    C --> F[Frame Extraction]
    C --> G[Embedding Generation]
    D --> H[Caption Generation]
    D --> I[Similarity Scoring]
    E --> J[Temporal Localization]
    F --> K[Video Storage]
    G --> L[Vector Database]
    H --> M[Text Processing]
    I --> N[Result Ranking]
    J --> O[Clip Extraction]
    
    subgraph "Frontend Layer"
        A
    end
    
    subgraph "Processing Pipeline"
        B
        C
        D
        E
    end
    
    subgraph "Core Services"
        F
        G
        H
        I
        J
    end
    
    subgraph "Storage Layer"
        K
        L
        M
    end
    
    subgraph "Output Layer"
        N
        O
    end
```

## 2. Technology Description

* **Frontend**: Python Streamlit/Gradio for web interface, OpenCV for video display

* **Backend**: Python FastAPI for API services, Celery for async processing

* **AI Models**: OpenCLIP (ViT-B/32), BLIP-2, UniVTG, Sentence Transformers

* **Video Processing**: Decord for frame extraction, FFmpeg for clip generation

* **Storage**: Local filesystem for videos, FAISS for vector similarity search

* **Dependencies**: PyTorch, Transformers, NumPy, Pandas

## 3. Route Definitions

| Route         | Purpose                                         |
| ------------- | ----------------------------------------------- |
| /             | Main interface for video upload and query input |
| /upload       | Video file upload and preprocessing status      |
| /process      | Processing pipeline execution and monitoring    |
| /results      | Timeline visualization and clip preview         |
| /api/upload   | REST API for video upload                       |
| /api/query    | REST API for event detection queries            |
| /api/status   | Processing status and progress tracking         |
| /api/download | Clip download and export functionality          |

## 4. API Definitions

### 4.1 Core API

**Video Upload**

```
POST /api/upload
```

Request:

| Param Name  | Param Type | isRequired | Description                     |
| ----------- | ---------- | ---------- | ------------------------------- |
| video\_file | file       | true       | Video file (MP4, AVI, MOV)      |
| video\_id   | string     | false      | Custom identifier for the video |

Response:

| Param Name   | Param Type | Description                          |
| ------------ | ---------- | ------------------------------------ |
| video\_id    | string     | Unique identifier for uploaded video |
| status       | string     | Upload status (success/error)        |
| duration     | float      | Video duration in seconds            |
| frame\_count | integer    | Total number of frames               |

**Event Detection Query**

```
POST /api/query
```

Request:

| Param Name | Param Type | isRequired | Description                             |
| ---------- | ---------- | ---------- | --------------------------------------- |
| video\_id  | string     | true       | Video identifier                        |
| query      | string     | true       | Natural language event description      |
| mode       | string     | false      | Processing mode (mvp/reranked/advanced) |
| top\_k     | integer    | false      | Number of top results to return         |
| threshold  | float      | false      | Confidence threshold (0.0-1.0)          |

Response:

| Param Name      | Param Type | Description                               |
| --------------- | ---------- | ----------------------------------------- |
| task\_id        | string     | Processing task identifier                |
| status          | string     | Task status (queued/processing/completed) |
| estimated\_time | integer    | Estimated completion time in seconds      |

**Results Retrieval**

```
GET /api/results/{task_id}
```

Response:

| Param Name            | Param Type | Description                |
| --------------------- | ---------- | -------------------------- |
| results               | array      | Array of detected events   |
| results\[].timestamp  | float      | Event timestamp in seconds |
| results\[].confidence | float      | Confidence score (0.0-1.0) |
| results\[].duration   | float      | Event duration in seconds  |
| results\[].clip\_url  | string     | URL to extracted clip      |

## 5. Server Architecture Diagram

```mermaid
graph TD
    A[FastAPI Application] --> B[Upload Controller]
    A --> C[Query Controller]
    A --> D[Results Controller]
    B --> E[Video Service]
    C --> F[Processing Service]
    D --> G[Export Service]
    E --> H[Frame Extractor]
    F --> I[Phase 1 Service]
    F --> J[Phase 2 Service]
    F --> K[Phase 3 Service]
    I --> L[OpenCLIP Model]
    J --> M[BLIP Model]
    K --> N[UniVTG Model]
    H --> O[(Video Storage)]
    L --> P[(Vector Database)]
    M --> P
    N --> P
    
    subgraph "API Layer"
        A
        B
        C
        D
    end
    
    subgraph "Service Layer"
        E
        F
        G
    end
    
    subgraph "Processing Layer"
        H
        I
        J
        K
    end
    
    subgraph "Model Layer"
        L
        M
        N
    end
    
    subgraph "Storage Layer"
        O
        P
    end
```

## 6. Data Model

### 6.1 Data Model Definition

```mermaid
erDiagram
    VIDEO ||--o{ FRAME : contains
    VIDEO ||--o{ QUERY : processes
    QUERY ||--o{ RESULT : generates
    RESULT ||--o{ CLIP : extracts
    FRAME ||--o{ EMBEDDING : has
    
    VIDEO {
        string video_id PK
        string filename
        float duration
        integer frame_count
        string format
        datetime uploaded_at
        string status
    }
    
    FRAME {
        string frame_id PK
        string video_id FK
        float timestamp
        integer frame_number
        string image_path
    }
    
    QUERY {
        string query_id PK
        string video_id FK
        string query_text
        string mode
        float threshold
        integer top_k
        datetime created_at
        string status
    }
    
    RESULT {
        string result_id PK
        string query_id FK
        float timestamp
        float confidence
        float duration
        string phase
        json metadata
    }
    
    CLIP {
        string clip_id PK
        string result_id FK
        float start_time
        float end_time
        string clip_path
        integer file_size
    }
    
    EMBEDDING {
        string embedding_id PK
        string frame_id FK
        vector embedding_vector
        string model_name
        datetime created_at
    }
```

### 6.2 Data Definition Language

**Video Table**

```sql
CREATE TABLE videos (
    video_id VARCHAR(36) PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    duration FLOAT NOT NULL,
    frame_count INTEGER NOT NULL,
    format VARCHAR(10) NOT NULL,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'uploaded'
);

CREATE INDEX idx_videos_status ON videos(status);
CREATE INDEX idx_videos_uploaded_at ON videos(uploaded_at DESC);
```

**Frame Table**

```sql
CREATE TABLE frames (
    frame_id VARCHAR(36) PRIMARY KEY,
    video_id VARCHAR(36) NOT NULL,
    timestamp FLOAT NOT NULL,
    frame_number INTEGER NOT NULL,
    image_path VARCHAR(500),
    FOREIGN KEY (video_id) REFERENCES videos(video_id)
);

CREATE INDEX idx_frames_video_id ON frames(video_id);
CREATE INDEX idx_frames_timestamp ON frames(timestamp);
```

**Query Table**

```sql
CREATE TABLE queries (
    query_id VARCHAR(36) PRIMARY KEY,
    video_id VARCHAR(36) NOT NULL,
    query_text TEXT NOT NULL,
    mode VARCHAR(20) DEFAULT 'mvp',
    threshold FLOAT DEFAULT 0.5,
    top_k INTEGER DEFAULT 10,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'queued',
    FOREIGN KEY (video_id) REFERENCES videos(video_id)
);

CREATE INDEX idx_queries_video_id ON queries(video_id);
CREATE INDEX idx_queries_status ON queries(status);
```

**Results Table**

```sql
CREATE TABLE results (
    result_id VARCHAR(36) PRIMARY KEY,
    query_id VARCHAR(36) NOT NULL,
    timestamp FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    duration FLOAT DEFAULT 5.0,
    phase VARCHAR(20) NOT NULL,
    metadata JSON,
    FOREIGN KEY (query_id) REFERENCES queries(query_id)
);

CREATE INDEX idx_results_query_id ON results(query_id);
CREATE INDEX idx_results_confidence ON results(confidence DESC);
CREATE INDEX idx_results_timestamp ON results(timestamp);
```

**Clips Table**

```sql
CREATE TABLE clips (
    clip_id VARCHAR(36) PRIMARY KEY,
    result_id VARCHAR(36) NOT NULL,
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    clip_path VARCHAR(500),
    file_size INTEGER,
    FOREIGN KEY (result_id) REFERENCES results(result_id)
);

CREATE INDEX idx_clips_result_id ON clips(result_id);
```

**Embeddings Table**

```sql
CREATE TABLE embeddings (
    embedding_id VARCHAR(36) PRIMARY KEY,
    frame_id VARCHAR(36) NOT NULL,
    embedding_vector BLOB NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (frame_id) REFERENCES frames(frame_id)
);

CREATE INDEX idx_embeddings_frame_id ON embeddings(frame_id);
CREATE INDEX idx_embeddings_model_name ON embeddings(model_name);
```

