# AdaptFL Server (FastAPI)

## Overview

The **AdaptFL Server** is a FastAPI-based backend that facilitates federated learning by managing **client authentication, model aggregation, and real-time updates** via WebSockets. The server securely stores model weights in **Azure Blob Storage**, aggregates model updates from edge devices, and notifies clients when a new global model is available.

## Features

- **Client Registration & Authentication**
- **Federated Model Aggregation (FedAvg, WFA)**
- **WebSocket Notifications for Model Updates**
- **Azure Blob Storage for Model Weights**
- **API Key-Based Secure Access**
- **Auto-Reconnection for Clients**

## API Endpoints

| Method   | Endpoint                   | Description                             |
| -------- | -------------------------- | --------------------------------------- |
| **POST** | `/register`                | Register a new client                   |
| **POST** | `/upload_weights`          | Upload model weights                    |
| **GET**  | `/get_latest_model`        | Download latest global model            |
| **POST** | `/aggregate-weights`        | Aggregate weights & update global model |
| **GET**  | `/get_data`          | Get detailed system information              |
| **WS**   | `/ws/{client_id}` | WebSocket for model update alerts       |

## Workflow

1. **Client connects** to the server’s WebSocket.
2. **Clients upload local model weights** to Azure Blob Storage.
3. **Server aggregates** weights and updates the global model.
4. **Notifies clients** when a new global model is available.
5. **Clients download** the latest model and continue training.

<br>

# AdaptFL Client (Edge Device)

🔗 [Go to AdaptFL Cleint Repository](https://github.com/UmarBalak/adaptfl_client)
