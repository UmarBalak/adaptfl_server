import os
import logging
import numpy as np
from typing import List, Optional, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from azure.storage.blob import BlobServiceClient, BlobClient
from tensorflow import keras
from datetime import datetime
import tempfile
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import asyncio
import re
import uuid
from fastapi import Body
from dotenv import load_dotenv
import logging
import os
import re
import tempfile
from typing import List, Optional, Tuple
import numpy as np
from azure.storage.blob import BlobServiceClient
import os
from fastapi.responses import HTMLResponse
import random
from sqlalchemy import create_engine, Column, String, DateTime, Table, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from datetime import datetime
import time


load_dotenv()


# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")  # Format: postgresql://user:password@host:port/database
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is missing")


global_vars = {
    'last_processed_timestamp': 0,
    'latest_version': 0
}

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Association table for many-to-many relationship between GlobalModel and Client
client_model_association = Table(
    'client_model_association', Base.metadata,
    Column('client_id', String, ForeignKey('clients.client_id')),
    Column('model_id', Integer, ForeignKey('global_models.id'))
)

# Client model
class Client(Base):
    __tablename__ = "clients"
    
    csn = Column(String, primary_key=True)
    client_id = Column(String, unique=True, nullable=False)
    api_key = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="Inactive")
    contribution_count = Column(Integer, default=0)

    models_contributed = relationship("GlobalModel", secondary=client_model_association, back_populates="clients")

# GlobalModel model
class GlobalModel(Base):
    __tablename__ = "global_models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(Integer, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    num_clients_contributed = Column(Integer, default=0)
    client_ids = Column(String) # Comma-seperated

    clients = relationship("Client", secondary=client_model_association, back_populates="models_contributed")

# Create tables
Base.metadata.create_all(bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()  # Commit changes before closing
    except Exception:
        db.rollback()  # Rollback on error
    finally:
        db.close()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket):

        await websocket.accept()
        self.active_connections[client_id] = websocket

    async def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def broadcast_model_update(self, message: str):
        disconnected_clients = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(message)
            except Exception as e:
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)

# Initialize FastAPI app with connection manager
manager = ConnectionManager()
app = FastAPI()

# Azure Blob Storage configuration
CLIENT_ACCOUNT_URL = os.getenv("CLIENT_ACCOUNT_URL")
SERVER_ACCOUNT_URL = os.getenv("SERVER_ACCOUNT_URL")
CLIENT_CONTAINER_NAME = os.getenv("CLIENT_CONTAINER_NAME")
SERVER_CONTAINER_NAME = os.getenv("SERVER_CONTAINER_NAME")
ARCH_BLOB_NAME = "model_architecture.keras"
CLIENT_NOTIFICATION_URL = os.getenv("CLIENT_NOTIFICATION_URL")

if not CLIENT_ACCOUNT_URL or not SERVER_ACCOUNT_URL:
    raise ValueError("Missing required environment variable: SAS url")


try:
    blob_service_client_client = BlobServiceClient(account_url=CLIENT_ACCOUNT_URL)
    blob_service_client_server = BlobServiceClient(account_url=SERVER_ACCOUNT_URL)
except Exception as e:
    raise



def get_model_architecture() -> Optional[object]:
    """
    Load model architecture from blob storage.
    """
    try:
        container_client = blob_service_client_client.get_container_client(CLIENT_CONTAINER_NAME)
        blob_client = container_client.get_blob_client(ARCH_BLOB_NAME)
        
        # Download architecture file to memory
        arch_data = blob_client.download_blob().readall()
        
        model = keras.models.load_model(arch_data, compile=False)

        return model
    
    except ImportError as e:
        print(f"ImportError: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None











def load_weights_from_blob(
    blob_service_client: BlobServiceClient,
    container_name: str,
    model,
    last_processed_timestamp: int
) -> Optional[Tuple[List[np.ndarray], List[dict], int]]:
    try:
        # Compile regex pattern to match blob names
        pattern = re.compile(r"client[0-9a-fA-F\-]+_v\d+_(\d{8}_\d{6})\.keras")
        container_client = blob_service_client.get_container_client(container_name)

        weights_list = []
        num_examples_list = []
        loss_list = []
        new_last_processed_timestamp = last_processed_timestamp

        blobs = list(container_client.list_blobs())
        
        for blob in blobs:
            match = pattern.match(blob.name)
            if match:
                timestamp_str = match.group(1)
                timestamp_int = int(timestamp_str.replace("_", ""))
                if timestamp_int > last_processed_timestamp:
                    blob_client = container_client.get_blob_client(blob.name)
                    
                    # Download the blob and load weights
                    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as temp_file:
                        download_stream = blob_client.download_blob()
                        temp_file.write(download_stream.readall())
                        temp_path = temp_file.name
                    
                    # Load weights into the model
                    model.load_weights(temp_path)
                    weights = model.get_weights()
                    
                    # Fetch metadata from the blob
                    blob_metadata = blob_client.get_blob_properties().metadata
                    if blob_metadata:
                        # Convert metadata values to appropriate types if necessary
                        num_examples = int(blob_metadata.get('num_examples', 0))
                        loss = float(blob_metadata.get('loss', 0.0))
                        if num_examples == 0:
                            continue  # Skip blobs with no valid 'num_examples' metadata
                        num_examples_list.append(num_examples)
                        loss_list.append(loss)

                    # Clean up temporary file
                    os.unlink(temp_path)

                    # Store weights and update timestamps
                    weights_list.append(weights)
                    new_last_processed_timestamp = max(new_last_processed_timestamp, timestamp_int)

        if not weights_list:
            return None, [], [], last_processed_timestamp
        
        return weights_list, num_examples_list, loss_list, new_last_processed_timestamp

    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return None, [], [], last_processed_timestamp


def save_weights_to_blob(weights: List[np.ndarray], filename: str, model) -> bool:
    """
    Save model weights to a blob.
    """
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as temp_file:
            temp_path = temp_file.name
            model.set_weights(weights)
            model.save_weights(temp_path)

        blob_client = blob_service_client_server.get_blob_client(
            container=SERVER_CONTAINER_NAME, 
            blob=filename
        )
        
        with open(temp_path, "rb") as file:
            blob_client.upload_blob(file, overwrite=True)
        
        return True
    except Exception as e:
        return False
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

def federated_weighted_averaging(weights_list, num_examples_list, loss_list, alpha=0.7):
    """Perform Weighted Federated Averaging with improved loss weighting."""
    if not weights_list or not num_examples_list or not loss_list:
        return None
    
    total_examples = sum(num_examples_list)
    if total_examples == 0:
        return None

    # Softmax-based loss weighting
    loss_weights = np.exp(-np.array(loss_list))
    loss_weights = loss_weights / np.sum(loss_weights)
    
    # Combine data size and loss weights
    final_weights = []
    for i in range(len(weights_list)):
        data_weight = num_examples_list[i] / total_examples
        combined_weight = alpha * data_weight + (1 - alpha) * loss_weights[i]
        final_weights.append(combined_weight)
    
    # Normalize final weights
    final_weights = np.array(final_weights) / np.sum(final_weights)
    
    # Initialize and compute weighted average
    avg_weights = [np.zeros_like(layer, dtype=np.float64) for layer in weights_list[0]]
    for i, layer_weights in enumerate(zip(*weights_list)):
        for client_idx, client_weights in enumerate(layer_weights):
            avg_weights[i] += client_weights * final_weights[client_idx]
            
    return avg_weights

def get_versioned_filename(version: int, prefix="g", extension="keras"):
    filename = f"{prefix}{version}.{extension}"
    return filename

def get_latest_model_version() -> str:
    # Fetch the latest model version from the database
    db = get_db()
    latest_model = db.query(GlobalModel).order_by(GlobalModel.version.desc()).first()
    
    if latest_model:
        return f"g{latest_model.version}.keras"
    return "none"


# Admin authentication function
def verify_admin(api_key: str):
    admin_key = os.getenv("ADMIN_API_KEY", "your_admin_secret_key")
    if api_key != admin_key:
        raise HTTPException(status_code=403, detail="Unauthorized admin access")

@app.on_event("startup")
async def startup_event():
    try:
        # Verify Azure Blob Storage connectivity
        container_client = blob_service_client_client.get_container_client(CLIENT_CONTAINER_NAME)
        blob_client = container_client.get_blob_client(ARCH_BLOB_NAME)
        
        if not blob_client.exists():
            logging.error(f"Model architecture file {ARCH_BLOB_NAME} not found in storage")
            raise FileNotFoundError(f"Model architecture file not found")
            
        logging.info("Successfully verified Azure Blob Storage connectivity and model file existence")
        
    except Exception as e:
        logging.error(f"Startup verification failed: {e}")
        raise

# @app.get("/")
# def read_root():
#     return {"message": "Hello, World!"}

import random

@app.get("/", response_class=HTMLResponse)
def read_root():
    quotes = [
        "The best way to predict the future is to code it.",
        "Simplicity is the soul of efficiency.",
        "Code is like humor. When you have to explain it, it’s bad.",
        "Programming isn’t about what you know; it’s about what you can figure out.",
        "Push yourself, because no one else is going to do it for you."
    ]
    quote = random.choice(quotes)
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Coder's Mood</title>
        <style>
            body {{
                background-color: #121212;
                color: #e0e0e0;
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                text-align: center;
            }}
            h1 {{
                font-size: 3em;
                margin-bottom: 0.5em;
            }}
            p {{
                font-size: 1.5em;
                font-style: italic;
            }}
            .quote {{
                font-size: 1.2em;
                color: #8f8f8f;
            }}
            .refresh-button {{
                background-color: #333;
                color: #fff;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 1em;
                transition: background-color 0.3s;
            }}
            .refresh-button:hover {{
                background-color: #555;
            }}
        </style>
    </head>
    <body>
        <div>
            <h1>Code, Create, Conquer</h1>
            <p>Keep pushing those limits!</p>
            <p class="quote">"{quote}"</p>
            <button class="refresh-button" onclick="window.location.reload();">Refresh Mood</button>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# Modified registration endpoint
@app.post("/register")
async def register(
    csn: str = Body(...),
    admin_api_key: str = Body(...),
    db: Session = Depends(get_db)
):
    verify_admin(admin_api_key)
    
    existing_client = db.query(Client).filter(Client.csn == csn).first()
    if existing_client:
        raise HTTPException(status_code=400, detail="Client already registered")
    
    client_id = str(uuid.uuid4())
    api_key = str(uuid.uuid4())
    
    new_client = Client(
        csn=csn,
        client_id=client_id,
        api_key=api_key
    )
    
    db.add(new_client)
    db.commit()
    
    return {
        "status": "success",
        "message": "Client registered successfully",
        "data": {"client_id": client_id, "api_key": api_key}
    }


@app.get("/aggregate-weights")
async def aggregate_weights():
    db = next(get_db())
    try:
        model = get_model_architecture()
        if not model:
            raise HTTPException(status_code=500, detail="Failed to load model architecture")

        weights_list, num_examples_list, loss_list, new_timestamp = load_weights_from_blob(
            blob_service_client_client, 
            CLIENT_CONTAINER_NAME, 
            model, 
            global_vars['last_processed_timestamp']
        )

        if not weights_list:
            return {"status": "no_update", "message": "No new weights found", "num_clients": 0}

        if not num_examples_list:
            return None

        global_vars['latest_version'] += 1
        filename = get_versioned_filename(global_vars['latest_version'])

        # avg_weights = federated_averaging(weights_list)
        avg_weights = federated_weighted_averaging(weights_list, num_examples_list, loss_list)
        if not avg_weights or not save_weights_to_blob(avg_weights, filename, model):
            raise HTTPException(status_code=500, detail="Failed to save aggregated weights")

        global_vars['last_processed_timestamp'] = new_timestamp
        
        # Update database
        client_ids = [c.client_id for c in db.query(Client).all()]
        new_model = GlobalModel(
            version=global_vars['latest_version'],
            num_clients_contributed=len(weights_list),
            client_ids=",".join(client_ids)
        )
        db.add(new_model)
        
        # Update contribution counts
        db.query(Client).update({"contribution_count": Client.contribution_count + 1})
        db.commit()

        await manager.broadcast_model_update(f"NEW_MODEL:{filename}")
        return {
            "status": "success",
            "message": f"Aggregated weights saved as {filename}",
            "num_clients": len(weights_list)
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{client_id}") 
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    db = SessionLocal()  # Direct session creation
    try:
        client = db.query(Client).filter(Client.client_id == client_id).first()
        if not client:
            await websocket.close(code=1008, reason="Unauthorized")
            return

        await manager.connect(client_id, websocket)
        client.status = "Active"
        db.commit()

        await websocket.send_text(f"Your status is now: {client.status}")

        latest_model_version = f"g{global_vars['latest_version']}.keras"
        await websocket.send_text(f"LATEST_MODEL:{latest_model_version}")
        
        while True:
            data = await websocket.receive_text()
            if not data:
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        await manager.disconnect(client_id)
        if client:
            client.status = "Inactive"
            db.commit()
        db.close()

# Scheduler setup
scheduler = BackgroundScheduler()

@scheduler.scheduled_job(CronTrigger(minute="*/2"))
def scheduled_aggregate_weights():
    """
    Scheduled task to aggregate weights every minute.
    """
    try:
        asyncio.run(aggregate_weights())
    except Exception as e:
        raise

scheduler.start()

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="localhost", port=8000)