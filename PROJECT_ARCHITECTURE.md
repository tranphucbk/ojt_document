# Kiến Trúc Dự Án — Generative AI Use Cases (GenU)

## 1. Tổng Quan

GenU là một ứng dụng full-stack chạy hoàn toàn trên AWS, được xây dựng như một tham chiếu kiến trúc (well-architected reference) cho việc tích hợp AI sinh tạo vào các nghiệp vụ thực tế. Toàn bộ infrastructure được định nghĩa bằng AWS CDK (TypeScript), frontend là React SPA phân phối qua CloudFront, và AI engine là Amazon Bedrock.

```
Người dùng
    │
    ▼
CloudFront (CDN)
    │
    ├─► S3 Bucket (React SPA — HTML/CSS/JS tĩnh)
    │
    └─► API Gateway (REST API, Cognito Authorizer)
              │
              ▼
         Lambda (Express Monolith — Lambda Web Adapter)
              │
              ├─► DynamoDB (lưu chat, message, use case)
              ├─► S3 (file upload, audio, transcript, video)
              └─► Amazon Bedrock (Claude, Nova, DeepSeek, ...)
```

---

## 2. Cấu Trúc Monorepo

Dự án dùng **npm workspaces**, tất cả package nằm trong `packages/`:

```
generative-ai-use-cases/
├── packages/
│   ├── web/                  ← React frontend (Vite + TypeScript)
│   ├── cdk/                  ← AWS CDK infrastructure + Lambda functions
│   ├── common/               ← Thư viện dùng chung (model metadata, filter utils)
│   ├── types/                ← TypeScript type definitions (dùng chung web ↔ lambda)
│   └── eslint-plugin-i18nhelper/ ← ESLint plugin tự viết, cấm hardcode tiếng Nhật
├── browser-extension/        ← Chrome extension (độc lập)
├── docs/                     ← Tài liệu MkDocs
├── setup-env.sh              ← Script lấy biến môi trường từ CloudFormation Output
├── docker-compose.dev.yml    ← Local dev với Docker
└── package.json              ← Root scripts (điều phối toàn bộ workspace)
```

**Package `types`** (`packages/types/`) export dưới tên `generative-ai-use-cases` — đây là nguồn sự thật duy nhất cho các kiểu dữ liệu dùng chung giữa frontend và Lambda (Chat, Message, Model, Agent, v.v.).

**Package `common`** (`packages/common/`) export dưới tên `@generative-ai-use-cases/common` — chứa metadata của từng model (capabilities/flags), danh sách model ID hợp lệ, và các util cho RAG filter.

---

## 3. Infrastructure — AWS CDK

### 3.1 Cấu hình tham số

Có hai lớp cấu hình, ưu tiên theo thứ tự:

```
parameter.ts  (cao hơn)
    └── cdk.json context  (mặc định)
```

- **`packages/cdk/cdk.json`**: Chứa toàn bộ giá trị mặc định. Đây là file chỉnh khi dùng CLI (`--context key=value`). Môi trường được chọn qua `context.env`.
- **`packages/cdk/parameter.ts`**: Cho phép override theo môi trường (dev/staging/prod). Dùng khi muốn quản lý nhiều env rõ ràng trong code.

### 3.2 Các Stack CDK

Tất cả stack được tạo trong `packages/cdk/lib/create-stacks.ts`. Mỗi stack tách biệt nhau vì lý do region hoặc vòng đời deploy:

| Stack | Mô tả | Region |
|-------|-------|--------|
| `GenerativeAiUseCasesStack` | Stack chính: Auth, API, Web, DB, các tính năng tùy chọn | Bất kỳ |
| `CloudFrontWafStack` | WAF với giới hạn IP/quốc gia. CloudFront **bắt buộc phải ở `us-east-1`** | `us-east-1` |
| `RagKnowledgeBaseStack` | Bedrock Knowledge Base + OpenSearch Serverless | `modelRegion` |
| `AgentStack` | Bedrock Agents (khi dùng cross-account Bedrock role) | Bất kỳ |
| `AgentCoreStack` | AgentCore Runtime (Generic / AgentBuilder) | `agentCoreRegion` |
| `ResearchAgentCoreStack` | AgentCore Runtime cho Research Agent | `agentCoreRegion` |
| `GuardrailStack` | Bedrock Guardrail | `modelRegion` |
| `DashboardStack` | CloudWatch Dashboard | Bất kỳ |
| `VideoTmpBucketStack` | S3 bucket tạm cho video generation (deploy theo region) | Nhiều region |
| `ClosedNetworkStack` | VPC, VPC Endpoint, NLB (chế độ mạng nội bộ) | Bất kỳ |

### 3.3 Các Construct trong GenerativeAiUseCasesStack

`packages/cdk/lib/construct/` — mỗi file là một Construct đóng gói một nhóm tài nguyên:

```
Auth           → Cognito UserPool + UserPoolClient + IdentityPool
               → Lambda trigger: kiểm tra email domain khi đăng ký
Database       → DynamoDB Table (chat/message) + StatsTable (token usage)
               → GlobalSecondaryIndex: FeedbackIndex
Api            → API Gateway REST API (Cognito Authorizer)
               → Lambda Monolith: apiHandler (Express + Lambda Web Adapter)
               → Lambda trực tiếp: predictStream, invokeFlow, optimizePrompt
               → S3 bucket: file upload (hình ảnh, tài liệu)
Web            → CloudFront Distribution → S3 Bucket (SPA)
               → Build frontend tại deploy time (NodejsBuild)
               → Inject VITE_APP_* env vars vào build
Rag            → Amazon Kendra Index + DataSource S3
               → EventBridge schedule (bật/tắt index theo lịch)
               → Step Functions (sync data)
Transcribe     → S3 bucket audio + transcript
               → Lambda: getTranscription
SpeechToSpeech → AppSync Event API (realtime WebSocket)
               → Lambda: speechToSpeechTask
McpApi         → Lambda (Docker image) + Function URL (streaming)
               → Dùng cho Model Context Protocol
AgentCore      → Gán quyền invoke tới Identity Pool cho các AgentCore runtime
UseCaseBuilder → DynamoDB Table (use case definitions + agent configs)
```

---

## 4. Luồng Hoạt Động Chính

### 4.1 Xác thực người dùng

```
Trình duyệt
    │ 1. Vào URL CloudFront → tải React SPA từ S3
    │
    │ 2. AWS Amplify (trong SPA) → Cognito UserPool
    │    Đăng nhập bằng email/password (hoặc SAML nếu bật)
    │
    │ 3. Cognito trả về: ID Token (JWT), Access Token, Refresh Token
    │
    │ 4. Amplify dùng ID Token + IdentityPool → lấy AWS temporary credentials
    │    (dùng để gọi Lambda trực tiếp và AppSync)
    │
    └─► Mọi request HTTP tiếp theo đều mang ID Token trong Authorization header
```

### 4.2 Chat thông thường (luồng streaming)

Đây là luồng quan trọng nhất, có hai đường đi tùy loại request:

**Đường 1: Streaming (mặc định cho chat)**

```
ChatPage (React)
    │
    │ useChat() hook → useChatApi().predictStream()
    │
    ▼
Frontend tạo LambdaClient dùng AWS SDK (với Cognito Identity credentials)
    │
    │ InvokeWithResponseStreamCommand → Lambda ARN: predictStreamFunction
    │ (Gọi thẳng Lambda, KHÔNG qua API Gateway)
    │
    ▼
Lambda: predictStream.ts
    │ handler = awslambda.streamifyResponse(...)
    │
    │ 1. Nhận PredictRequest { model, messages, idToken }
    │ 2. Xác thực idToken qua aws-jwt-verify (verifyToken)
    │ 3. Gọi api[model.type].invokeStream() → bedrockApi / bedrockKbApi / ...
    │
    ▼
Amazon Bedrock
    │ ConverseStreamCommand → Claude / Nova / DeepSeek / ...
    │ Trả về token stream
    │
    ▼
Lambda stream → Frontend nhận từng token → render dần lên màn hình
```

**Đường 2: REST API (CRUD, không streaming)**

```
Frontend (useHttp hook — axios)
    │ Authorization: <ID Token>
    │
    ▼
API Gateway (REST) → Cognito UserPool Authorizer
    │ Authorizer xác thực JWT, inject claims vào requestContext
    │
    ▼
Lambda Monolith: apiHandler (Express app)
    │ Lambda Web Adapter (LWA) nhận HTTP request
    │
    │ Express Router:
    │   POST /chats        → createChat.ts
    │   GET  /chats        → listChats.ts
    │   GET  /chats/:id/messages → listMessages.ts
    │   POST /chats/:id/messages → createMessages.ts
    │   ...
    │
    ▼
Business Logic Handler (vd: createChat.ts)
    │ userId = event.requestContext.authorizer.claims['cognito:username']
    │
    ▼
repository.ts (DynamoDB Document Client)
    │ PutCommand / QueryCommand / UpdateCommand / DeleteCommand
    │
    ▼
DynamoDB Table
```

### 4.3 Lưu lịch sử hội thoại

Sau khi streaming xong, frontend lưu tin nhắn vào DynamoDB:

```
Frontend (useChat hook)
    │
    │ 1. predictStream() → nhận toàn bộ response
    │ 2. POST /chats/:id/messages → createMessages.ts
    │    Body: { conversation: [userMsg, assistantMsg], usecase, title }
    │
    ▼
createMessages.ts → repository.ts
    │ BatchWriteCommand → ghi nhiều message cùng lúc
    │ Đồng thời cập nhật Stats (token usage) vào StatsTable
    │
    ▼
DynamoDB
```

### 4.4 RAG Chat (Kendra hoặc Knowledge Base)

```
RagPage hoặc RagKnowledgeBasePage
    │
    │ useRag() → useChatApi().predictStream()
    │ model.type = 'bedrockKb' (Knowledge Base) hoặc 'bedrock' + Kendra tool
    │
    ▼
Lambda: predictStream.ts
    │ model.type === 'bedrockKb' → bedrockKbApi.invokeStream()
    │
    ▼
bedrockKbApi.ts
    │ RetrieveAndGenerateStreamCommand
    │   → Knowledge Base ID (OpenSearch Serverless vector store)
    │   → Embedding model (Titan / Cohere)
    │   → Reranking model (nếu bật)
    │   → LLM để tổng hợp câu trả lời
    │
    ▼
Bedrock Knowledge Base
    │ 1. Embed câu hỏi → vector
    │ 2. Tìm kiếm trong OpenSearch Serverless
    │ 3. Trả về chunks liên quan + source references
    │ 4. LLM tổng hợp câu trả lời + trích dẫn nguồn
    │
    ▼
Frontend nhận stream + hiển thị nguồn tham chiếu
```

### 4.5 Voice Chat (Speech-to-Speech)

```
VoiceChatPage
    │
    │ useTranslation + microphone input
    │
    ▼
AppSync Event API (WebSocket / realtime)
    │ Auth: Cognito UserPool Token
    │
    ▼
Lambda: speechToSpeechTask
    │ Bedrock: Nova Sonic (amazon.nova-sonic-v1:0)
    │ Bidirectional audio streaming
    │
    ▼
Frontend nhận audio stream → phát lại cho người dùng
```

### 4.6 MCP Chat (Model Context Protocol)

```
McpChatPage
    │
    │ useMcp() → gọi Lambda Function URL trực tiếp
    │ Auth: AWS SigV4 (Cognito Identity credentials)
    │
    ▼
Lambda (Docker image): mcp-api/
    │ Lambda Web Adapter (streaming mode)
    │ Chạy MCP server trong container
    │
    ▼
Bedrock + các MCP tools
```

---

## 5. Frontend Architecture

### 5.1 Cấu trúc thư mục `packages/web/src/`

```
App.tsx              ← Router root, điều phối navigation dựa trên feature flags
pages/               ← Một file = một route (ChatPage, RagPage, GenerateImagePage, ...)
components/          ← UI components tái sử dụng
hooks/               ← Custom hooks (chia 2 loại: API hook và business logic hook)
i18n/                ← Cấu hình i18next (YAML-based, hỗ trợ en/ja/th/zh/vi/ko)
prompts/             ← Template prompt cho từng use case
utils/               ← Helper functions
assets/              ← SVG, hình ảnh được bundle
@types/              ← Type definitions riêng cho frontend
public/locales/      ← File dịch YAML (translation/prompts) theo ngôn ngữ
```

### 5.2 Phân tách hook — Nguyên tắc quan trọng

```
use{Feature}Api.ts    ← CHỈ giao tiếp với server (axios, LambdaClient)
                         Không chứa state, không chứa logic UI
use{Feature}.ts       ← Business logic, state management
                         Gọi API hook, dùng Zustand store
Page component        ← Kết hợp hooks, render UI
```

**Ví dụ cụ thể (luồng Chat):**

```
useChatApi.ts
  ├── predictStream()    → gọi Lambda trực tiếp (streaming)
  ├── createChat()       → POST /chats
  ├── createMessages()   → POST /chats/:id/messages
  ├── listMessages()     → GET /chats/:id/messages (SWR cache)
  └── listChats()        → GET /chats (SWR infinite scroll)

useChat.ts (Zustand store)
  ├── state: { chats, messages, loading, writing, modelIds }
  ├── init()             → khởi tạo chat session
  ├── post()             → gọi predictStream + lưu message
  ├── edit()             → chỉnh sửa tin nhắn và regenerate
  └── restore()          → load lại chat cũ từ DynamoDB

ChatPage.tsx
  ├── useChat(pathname)  → lấy state và actions
  ├── useChatApi()       → lấy createChat, listMessages
  └── render ChatMessage, InputChatContent, ...
```

### 5.3 HTTP layer — useHttp

`useHttp.ts` là adapter duy nhất wrapping axios:
- **GET** → dùng **SWR** (cache, revalidate tự động)
- **GET phân trang** → dùng **SWR Infinite**
- **POST/PUT/DELETE** → axios thuần
- Interceptor tự động đính kèm **Cognito ID Token** vào mọi request

### 5.4 Feature Flags (biến môi trường Vite)

Feature flags được inject vào lúc build (qua `VITE_APP_*`) và được lấy từ CloudFormation Output bởi `setup-env.sh`. App.tsx đọc chúng để ẩn/hiện menu và route:

| Flag | Tính năng |
|------|-----------|
| `VITE_APP_RAG_ENABLED` | Kendra RAG |
| `VITE_APP_RAG_KNOWLEDGE_BASE_ENABLED` | Bedrock Knowledge Base RAG |
| `VITE_APP_AGENT_ENABLED` | Bedrock Agents |
| `VITE_APP_MCP_ENABLED` | MCP Chat |
| `VITE_APP_AGENT_CORE_ENABLED` | AgentCore |
| `VITE_APP_RESEARCH_AGENT_ENABLED` | Research Agent |
| `VITE_APP_USE_CASE_BUILDER_ENABLED` | Use Case Builder |
| `VITE_APP_HIDDEN_USE_CASES` | Ẩn các use case cụ thể |

---

## 6. Backend Architecture — Lambda

### 6.1 Express Monolith (API Gateway path)

```
packages/cdk/lambda/
├── api/
│   ├── index.ts         ← Express app entry, CORS, middleware, routing
│   ├── run.sh           ← Script khởi động Lambda Web Adapter
│   └── routes/
│       ├── helpers.ts   ← wrapHandler: adapter tạm thời (APIGatewayEvent → Express)
│       ├── chats.ts     ← /chats, /chats/:id, /chats/:id/messages
│       ├── predict.ts   ← /predict (non-streaming)
│       ├── rag.ts       ← /rag (Kendra)
│       ├── ragKnowledgeBase.ts ← /rag-knowledge-base
│       ├── image.ts     ← /image/generate
│       ├── video.ts     ← /video/*
│       ├── transcribe.ts ← /transcribe/*
│       ├── systemContexts.ts ← /system-contexts
│       ├── shares.ts    ← /share-id
│       ├── tokenUsage.ts ← /token-usage
│       ├── agentBuilder.ts ← /agent-builder
│       └── ...
```

**Cơ chế `wrapHandler`**: Do các handler được viết theo kiểu `APIGatewayProxyEvent` cũ, `helpers.ts` có adapter tạm thời chuyển đổi Express `req/res` thành `APIGatewayProxyEvent` để tái sử dụng code. Kế hoạch là migrate dần sang Express-native handlers.

### 6.2 Business Logic handlers

Mỗi file ở `lambda/` là một API endpoint:

```
createChat.ts     → userId từ JWT claims → repository.createChat()
listChats.ts      → QueryCommand theo userId
findChatById.ts   → QueryCommand theo userId + chatId
deleteChat.ts     → TransactWriteCommand (xóa chat + toàn bộ messages)
createMessages.ts → BatchWriteCommand (ghi nhiều tin nhắn + update stats)
listMessages.ts   → QueryCommand theo chatId
generateImage.ts  → bedrockApi.invokeImageGeneration()
generateVideo.ts  → bedrockApi.invokeVideoGeneration() → S3
getTranscription.ts → Transcribe → S3 → DynamoDB
```

### 6.3 Data Access Layer — repository.ts

`repository.ts` là lớp duy nhất tương tác trực tiếp với DynamoDB. Dùng `DynamoDBDocumentClient` (AWS SDK v3).

**Schema DynamoDB (Main Table)**:
```
Partition Key: id       = "user#<userId>"
Sort Key:      createdDate

Dữ liệu Chat:
  id            = "user#<userId>"
  createdDate   = timestamp
  chatId        = "chat#<uuid>"
  usecase       = tên use case
  title         = tiêu đề chat
  updatedDate   = timestamp

Dữ liệu Message:
  id            = "chat#<chatId>"
  createdDate   = timestamp
  messageId     = uuid
  role          = "user" | "assistant" | "system"
  content       = nội dung text
  feedback      = "good" | "bad" | ""
  usecase       = tên use case
  userId        = <userId>
  extraData     = [...] (file, image refs)
```

**Stats Table** (token usage):
```
Partition Key: id      = "YYYY-MM" (tháng)
Sort Key:      userId
```

### 6.4 Lambda trực tiếp (không qua API Gateway)

```
predictStream.ts     ← Streaming chat, gọi từ frontend qua LambdaClient
invokeFlow.ts        ← Bedrock Flows (workflow AI)
copyVideoJob.ts      ← Copy video từ S3 tmp sang bucket chính
optimizePrompt.ts    ← Tối ưu prompt trước khi gửi
```

Các Lambda này được **invoke trực tiếp** từ frontend bằng AWS SDK với credentials từ Cognito Identity Pool — không qua API Gateway, không có Cognito Authorizer. Xác thực thực hiện thủ công bên trong Lambda bằng `verifyToken()` (aws-jwt-verify).

---

## 7. Bedrock Integration

### 7.1 Model types và routing

Mỗi `Model` object có `type` quyết định backend nào được dùng:

```typescript
type Model = {
  type: 'bedrock' | 'bedrockAgent' | 'bedrockKb' | 'sagemaker';
  modelId: string;
  region?: string;       // Override MODEL_REGION nếu model ở region khác
  sessionId?: string;    // Dùng cho agent sessions
};
```

```
api.ts
├── bedrock      → bedrockApi.ts      (ConverseAPI, image, video)
├── bedrockAgent → bedrockAgentApi.ts (Bedrock Agents runtime)
├── bedrockKb    → bedrockKbApi.ts    (RetrieveAndGenerate)
└── sagemaker    → sagemakerApi.ts    (SageMaker endpoints)
```

### 7.2 Multi-region support

Mỗi model có thể chỉ định `region` riêng — `bedrockClient.ts` tạo `BedrockRuntimeClient` với region tương ứng. Điều này cho phép dùng cross-region inference profiles (prefix `us.`, `eu.`, `global.`).

### 7.3 Model metadata

`packages/common/src/application/model.ts` chứa `modelMetadata`: map từ `modelId` → `{ displayName, flags }`. Flags quyết định capabilities:

```typescript
flags: {
  text, doc, image, video,   // input capabilities
  image_gen, video_gen,      // output capabilities
  reasoning, adaptiveThinking,
  speechToSpeech,
  light,                     // dùng cho lightweight tasks (title generation)
  legacy                     // model cũ, vẫn hỗ trợ
}
```

---

## 8. Các Tính Năng Đặc Biệt

### 8.1 Use Case Builder

Cho phép người dùng tạo custom AI use case không cần code:
- Lưu trong `UseCaseBuilderTable` (DynamoDB riêng)
- Mỗi use case là một prompt template + system context + model config
- `packages/cdk/lambda/useCaseBuilder/` chứa toàn bộ CRUD logic

### 8.2 AgentCore (AgentBuilder)

Tính năng cao cấp để build agent:
- **AgentCoreStack**: Deploy AgentCore Runtime (separate stack vì có lifecycle riêng)
- Frontend dùng `@aws-sdk/client-bedrock-agentcore` để invoke runtime trực tiếp
- Identity Pool cần quyền `bedrock-agentcore:InvokeAgentRuntime`

### 8.3 Closed Network Mode

Chế độ triển khai trong VPC nội bộ (không có internet):
- **ClosedNetworkStack**: VPC, Private Subnets, VPC Endpoints cho tất cả AWS services
- API Gateway dùng `PRIVATE` endpoint type với VPC Endpoint restriction
- CloudFront được thay bằng NLB + Route53 nội bộ

### 8.4 Internationalization (i18n)

- Translation files: `packages/web/public/locales/translation/{lang}.yaml`
- Prompt files: `packages/web/public/locales/prompts/{lang}.yaml`
- 6 ngôn ngữ: `en`, `ja`, `th`, `zh`, `vi`, `ko`
- ESLint plugin `i18nhelper` tự viết — báo lỗi khi có hardcode tiếng Nhật trong source code
- Ngôn ngữ fallback: `en`

### 8.5 File Upload

```
Frontend upload file → useFileApi.ts
    │ 1. GET /files/upload-url → Lambda → S3 presigned URL
    │ 2. PUT thẳng lên S3 (presigned URL, bypass API Gateway)
    │ 3. Gửi s3Url trong ExtraData của message
    │
    ▼
Lambda predictStream nhận ExtraData
    │ source.type = 's3' → chuyển thành base64 hoặc S3 reference cho Bedrock
    │
    ▼
Bedrock ConverseAPI nhận document/image từ S3 hoặc base64
```

---

## 9. Local Development

### 9.1 Chế độ thông thường (sau khi deploy lên AWS)

```bash
npm run web:devw    # Mac/Linux: tự lấy env từ CloudFormation Output
npm run web:devww   # Windows PowerShell
```

`setup-env.sh` chạy `aws cloudformation describe-stacks` → parse các Output → export thành `VITE_APP_*` env vars → khởi động Vite dev server.

### 9.2 Chế độ Docker (full-stack local)

```bash
npm run dev:docker --env=<env>
```

Chạy hai container:
- **`api`**: Express Lambda Monolith dùng `npx tsx watch` (hot reload) — port 8080
- **`web`**: Vite dev server — port 5173

Container `api` mount `~/.aws:/root/.aws` để dùng AWS credentials local và gọi thẳng DynamoDB, S3, Bedrock trên AWS.

---

## 10. Deploy Flow

```
npm run cdk:deploy
    │
    ├── packages/cdk/bin/generative-ai-use-cases.ts
    │     └── getParams(app) → create-stacks.ts → tạo tất cả Stack objects
    │
    ├── Web Construct:
    │     NodejsBuild → chạy `vite build` với VITE_APP_* từ CloudFormation Output
    │     → Upload artifact lên S3 → CloudFront invalidation
    │
    └── CloudFormation deploy toàn bộ stacks
          → Output: ApiEndpoint, UserPoolId, ModelIds, ... (dùng bởi setup-env.sh)
```

Build frontend xảy ra **tại thời điểm CDK deploy** (không phải CI riêng). `NodejsBuild` construct từ `@cdklabs/deploy-time-build` chạy `vite build` trong CodeBuild ephemeral và upload kết quả lên S3.
