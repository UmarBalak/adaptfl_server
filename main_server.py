import os
import logging
import numpy as np
import time
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
from sqlalchemy import func
import uuid
from fastapi import Body
from fastapi.responses import JSONResponse
import logging
import os
import re
import tempfile
from typing import List, Optional, Tuple
import numpy as np
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

import os
from sqlalchemy import create_engine, Column, String, DateTime, Table, Integer, ForeignKey, select
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime

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

# Example SQLAlchemy model for Global Variables
class GlobalVars(Base):
    __tablename__ = "global_vars"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    value = Column(String)  # Can store timestamps, latest version, etc.


try:
    # Create tables
    Base.metadata.create_all(bind=engine)
except Exception as e:
    logging.error(f"Failed to create tables: {e}")


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


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket):

        await websocket.accept()
        self.active_connections[client_id] = websocket

        logging.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")

    async def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logging.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast_model_update(self, message: str):
        disconnected_clients = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(message)
                logging.info(f"Update notification sent to client {client_id}")
            except Exception as e:
                logging.error(f"Failed to send update to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)

# Azure Blob Storage configuration
CLIENT_ACCOUNT_URL = os.getenv("CLIENT_ACCOUNT_URL")
SERVER_ACCOUNT_URL = os.getenv("SERVER_ACCOUNT_URL")
CLIENT_CONTAINER_NAME = os.getenv("CLIENT_CONTAINER_NAME")
SERVER_CONTAINER_NAME = os.getenv("SERVER_CONTAINER_NAME")
ARCH_BLOB_NAME = "model_architecture.keras"
CLIENT_NOTIFICATION_URL = os.getenv("CLIENT_NOTIFICATION_URL")

if not CLIENT_ACCOUNT_URL or not SERVER_ACCOUNT_URL:
    logging.error("SAS url environment variable is missing.")
    raise ValueError("Missing required environment variable: SAS url")


try:
    blob_service_client_client = BlobServiceClient(account_url=CLIENT_ACCOUNT_URL)
    blob_service_client_server = BlobServiceClient(account_url=SERVER_ACCOUNT_URL)
except Exception as e:
    logging.error(f"Failed to initialize Azure Blob Service: {e}")
    raise



def get_model_architecture() -> Optional[object]:
    """
    Load model architecture from blob storage.
    """
    try:
        container_client = blob_service_client_client.get_container_client(CLIENT_CONTAINER_NAME)
        logging.info("Container client initialized successfully.")
        blob_client = container_client.get_blob_client(ARCH_BLOB_NAME)
        logging.info("Blob client initialized successfully.")
        
        # Download architecture file to memory
        arch_data = blob_client.download_blob().readall()
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as temp_file:
            temp_file.write(arch_data)
            temp_path = temp_file.name
        
        model = keras.models.load_model(temp_path, compile=False)

        os.unlink(temp_path)
        return model
    
    except ImportError as e:
        logging.error(f"Import error while loading model architecture: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return None
    except Exception as e:
        logging.error(f"Error loading model architecture: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
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
                if timestamp_int > int(last_processed_timestamp):
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
            logging.info(f"No new weights found since {last_processed_timestamp}.")
            return None, [], [], last_processed_timestamp

        logging.info(f"Loaded weights from {len(weights_list)} files.")
        return weights_list, num_examples_list, loss_list, new_last_processed_timestamp

    except Exception as e:
        logging.error(f"Error loading weights: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return None, [], [], last_processed_timestamp

# Load the last processed timestamp from the database
def load_last_processed_timestamp(db):
    timestamp = db.query(GlobalVars).filter_by(key="last_processed_timestamp").first()
    return timestamp.value if timestamp else None

# Save the last processed timestamp to the database
def save_last_processed_timestamp(db, new_timestamp):
    timestamp_record = db.query(GlobalVars).filter_by(key="last_processed_timestamp").first()
    if timestamp_record:
        timestamp_record.value = new_timestamp
    else:
        # If the timestamp doesn't exist, insert a new record
        new_record = GlobalVars(key="last_processed_timestamp", value=new_timestamp)
        db.add(new_record)
    db.commit()


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
        
        logging.info(f"Successfully saved weights to blob: {filename}")
        return True
    except Exception as e:
        logging.error(f"Error saving weights to blob: {e}")
        return False
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

def federated_weighted_averaging(weights_list, num_examples_list, loss_list, alpha=0.7):
    """Perform Weighted Federated Averaging with improved loss weighting."""
    if not weights_list or not num_examples_list or not loss_list:
        logging.error("Missing inputs for aggregation.")
        return None
    
    total_examples = sum(num_examples_list)
    if total_examples == 0:
        logging.error("Total examples is zero.")
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

import numpy as np

def add_laplace_noise(weights, epsilon, sensitivity):
    """
    Adds Laplace noise to the model weights for differential privacy.
    - weights: model weights to perturb.
    - epsilon: privacy budget.
    - sensitivity: sensitivity of the weights.
    """
    noise = np.random.laplace(0, sensitivity / epsilon, size=weights.shape)
    return weights + noise

def federated_weighted_averaging_with_dp(weights_list, num_examples_list, loss_list, epsilon=0.1, sensitivity=1.0, alpha=0.7):
    """Perform Weighted Federated Averaging with DP noise addition to the aggregated weights."""
    if not weights_list or not num_examples_list or not loss_list:
        logging.error("Missing inputs for aggregation.")
        return None
    
    total_examples = sum(num_examples_list)
    if total_examples == 0:
        logging.error("Total examples is zero.")
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

    # Convert to numpy array for easier manipulation
    avg_weights = np.array(avg_weights)

    # Add DP noise to the aggregated weights
    dp_weights = []
    for weights in avg_weights:
        # Apply Laplace noise to each layer's weights
        dp_weights.append(add_laplace_noise(weights, epsilon, sensitivity))
    
    return dp_weights


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

# Initialize FastAPI app with connection manager
manager = ConnectionManager()
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "HELLO, WORLD. Welcome to the AdaptFL Server!"}

@app.get("/health", response_class=JSONResponse)
async def health_check():
    """
    Health check endpoint to verify service availability.
    """
    return {"status": "healthy", "message": "The application is running successfully."}

@app.get("/get_data/")
async def get_all_data(db: Session = Depends(get_db)):
    try:
        # Query all data from the 'clients' table
        clients = db.execute(select(Client)).scalars().all()
        
        # Query all data from the 'global_models' table
        global_models = db.execute(select(GlobalModel)).scalars().all()
        
        # Query all data from the 'global_vars' table
        global_vars = db.execute(select(GlobalVars)).scalars().all()

        # Return all the data as a dictionary
        return {
            "clients": clients,
            "global_models": global_models,
            "global_vars": global_vars
        }
    except Exception as e:
        return {"error": str(e)}


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



# @app.get("/aggregate-weights")
# async def aggregate_weights():
#     db = next(get_db())
#     try:
#         # Load last processed timestamp from the database
#         last_processed_timestamp = load_last_processed_timestamp(db)
#         global_vars['last_processed_timestamp'] = last_processed_timestamp or 0  # Use 0 if no timestamp is found

#         model = get_model_architecture()
#         if not model:
#             raise HTTPException(status_code=500, detail="Failed to load model architecture")

#         weights_list, num_examples_list, loss_list, new_timestamp = load_weights_from_blob(
#             blob_service_client_client, 
#             CLIENT_CONTAINER_NAME, 
#             model, 
#             global_vars['last_processed_timestamp']
#         )

#         if not weights_list:
#             return {"status": "no_update", "message": "No new weights found", "num_clients": 0}
#         if len(weights_list) < 2:
#             return {"status": "no_update", "message": "Only 1 weight file found", "num_clients": 1}
#         if not num_examples_list:
#             logging.error("Example counts missing for aggregation")
#             return None

#         # global_vars['latest_version'] += 1
#         # Synchronize latest version from the database
#         latest_model = db.query(GlobalModel).order_by(GlobalModel.version.desc()).first()
#         global_vars['latest_version'] = latest_model.version if latest_model else 0

#         # Increment version
#         global_vars['latest_version'] += 1
#         # Ensure no duplicate version
#         if db.query(GlobalModel).filter_by(version=global_vars['latest_version']).first():
#             raise HTTPException(status_code=409, detail=f"Model with version {global_vars['latest_version']} already exists")

#         filename = get_versioned_filename(global_vars['latest_version'])

#         logging.info(f"Aggregating weights from {len(weights_list)} clients")
#         # epsilon = 0.5  # Privacy budget
#         # sensitivity = 1.0  # Sensitivity
#         # avg_weights = federated_weighted_averaging(weights_list, num_examples_list, loss_list, epsilon, sensitivity)
#         avg_weights = federated_weighted_averaging(weights_list, num_examples_list, loss_list)
#         logging.info(f"Aggregation completed.")
#         if not avg_weights or not save_weights_to_blob(avg_weights, filename, model):
#             raise HTTPException(status_code=500, detail="Failed to save aggregated weights")

#         # Save the new timestamp to the database
#         save_last_processed_timestamp(db, new_timestamp)

#         # Update the database with model and client information
#         client_ids = [c.client_id for c in db.query(Client).all()]
#         new_model = GlobalModel(
#             version=global_vars['latest_version'],
#             num_clients_contributed=len(weights_list),
#             client_ids=",".join(client_ids)
#         )
#         db.add(new_model)
        
#         # Get the client IDs of the contributing clients
#         contributing_client_ids = [client.client_id for client in db.query(Client).filter(Client.client_id.in_(client_ids)).all()]

#         # Update contribution counts for only the contributing clients
#         db.query(Client).filter(Client.client_id.in_(contributing_client_ids)).update(
#             {"contribution_count": Client.contribution_count + 1},
#             synchronize_session=False  # To avoid unnecessary session flush
#         )
#         db.commit()

#         await manager.broadcast_model_update(f"NEW_MODEL:{filename}")
#         return {
#             "status": "success",
#             "message": f"Aggregated weights saved as {filename}",
#             "num_clients": len(weights_list)
#         }
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/aggregate-weights")
async def aggregate_weights():
    db = next(get_db())
    try:
        # Load last processed timestamp from the database
        last_processed_timestamp = load_last_processed_timestamp(db)
        global_vars['last_processed_timestamp'] = last_processed_timestamp or 0  # Use 0 if no timestamp is found

        model = get_model_architecture()
        if not model:
            raise HTTPException(status_code=500, detail="Failed to load model architecture")

        weights_list, num_examples_list, loss_list, new_timestamp, contributing_client_ids = load_weights_from_blob(
            blob_service_client_client, 
            CLIENT_CONTAINER_NAME, 
            model, 
            global_vars['last_processed_timestamp']
        )

        if not weights_list:
            return {"status": "no_update", "message": "No new weights found", "num_clients": 0}
        if len(weights_list) < 2:
            return {"status": "no_update", "message": "Only 1 weight file found", "num_clients": 1}
        if not num_examples_list:
            print("Example counts missing for aggregation")
            logging.error("Example counts missing for aggregation")
            return None

        # Synchronize latest version from the database
        latest_model = db.query(GlobalModel).order_by(GlobalModel.version.desc()).first()
        global_vars['latest_version'] = latest_model.version if latest_model else 0

        # Increment version
        global_vars['latest_version'] += 1
        # Ensure no duplicate version
        if db.query(GlobalModel).filter_by(version=global_vars['latest_version']).first():
            raise HTTPException(status_code=409, detail=f"Model with version {global_vars['latest_version']} already exists")

        filename = get_versioned_filename(global_vars['latest_version'])

        logging.info(f"Aggregating weights from {len(weights_list)} clients")
        avg_weights = federated_weighted_averaging(weights_list, num_examples_list, loss_list)
        logging.info(f"Aggregation completed.")
        if not avg_weights or not save_weights_to_blob(avg_weights, filename, model):
            raise HTTPException(status_code=500, detail="Failed to save aggregated weights")

        # Save the new timestamp to the database
        save_last_processed_timestamp(db, new_timestamp)

        # Only update contribution count for the contributing clients
        db.query(Client).filter(Client.client_id.in_(contributing_client_ids)).update(
            {"contribution_count": Client.contribution_count + 1},
            synchronize_session=False  # To avoid unnecessary session flush
        )
        db.commit()

        # Update the database with model and client information, saving only contributing client IDs
        new_model = GlobalModel(
            version=global_vars['latest_version'],
            num_clients_contributed=len(contributing_client_ids),
            client_ids=",".join(contributing_client_ids)  # Only contributing clients
        )
        db.add(new_model)
        db.commit()

        await manager.broadcast_model_update(f"NEW_MODEL:{filename}")
        return {
            "status": "success",
            "message": f"Aggregated weights saved as {filename}",
            "num_clients": len(contributing_client_ids)
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{client_id}") 
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    db = SessionLocal()  # Direct session creation
    try:
        # Check if the client exists in the database
        client = db.query(Client).filter(Client.client_id == client_id).first()
        if not client:
            await websocket.close(code=1008, reason="Unauthorized")
            logging.warning(f"Unauthorized access attempt by client {client_id}.")
            return

        # Add the client to the WebSocket manager and update status
        await manager.connect(client_id, websocket)
        client.status = "Active"
        db.commit()
        logging.info(f"Client {client_id} connected successfully, status updated to 'Active'.")

        # Inform the client about their updated status
        await websocket.send_text(f"Your status is now: {client.status}")

        latest_model = db.query(GlobalModel).order_by(GlobalModel.version.desc()).first()
        global_vars['latest_version'] = latest_model.version if latest_model else 0

        # Send the latest model version to the client
        latest_model_version = f"g{global_vars['latest_version']}.keras"
        await websocket.send_text(f"LATEST_MODEL:{latest_model_version}")

        while True:
            try:
                # Receive and handle messages from the client
                data = await websocket.receive_text()

                if not data:
                    break

                # Dynamically fetch the client's updated status and send updates if needed
                retry_attempts = 3
                for attempt in range(retry_attempts):
                    try:
                        updated_client = db.query(Client).filter(Client.client_id == client_id).first()
                        if updated_client and updated_client.status != client.status:
                            client.status = updated_client.status
                            db.commit()
                            await websocket.send_text(f"Your updated status is: {client.status}")
                        break
                    except SQLAlchemyError as db_error:
                        logging.error(f"Attempt {attempt + 1} - Database error for client {client_id}: {db_error}", exc_info=True)
                        if attempt < retry_attempts - 1:
                            time.sleep(2)  # Wait before retrying
                        else:
                            raise

            except WebSocketDisconnect:
                logging.info(f"Client {client_id} disconnected gracefully.")
                break  # Exit loop on graceful disconnect
            except Exception as inner_error:
                logging.error(f"Error handling message from client {client_id}: {inner_error}", exc_info=True)
                await websocket.send_text("An error occurred. Please try again later.")
                break

    except SQLAlchemyError as db_error:
        # Handle specific database errors
        logging.error(f"Database error for client {client_id}: {db_error}", exc_info=True)
        await websocket.close(code=1002, reason="Database error.")
    except WebSocketDisconnect:
        logging.info(f"Client {client_id} disconnected unexpectedly.")
    except HTTPException as http_error:
        # Catch HTTP exceptions (like 400, 404 errors) if needed
        logging.error(f"HTTP error for client {client_id}: {http_error}", exc_info=True)
    except Exception as e:
        # Catch all other exceptions
        logging.error(f"Unexpected error for client {client_id}: {e}", exc_info=True)
        await websocket.close(code=1000, reason="Internal server error.")
    finally:
        # Cleanup: Disconnect the client and update the database
        try:
            await manager.disconnect(client_id)
            # Update status to "Inactive"
            client = db.query(Client).filter(Client.client_id == client_id).first()
            if client:
                client.status = "Inactive"
                db.commit()
                logging.info(f"Client {client_id} is now inactive. Db updated successfully!")
        except SQLAlchemyError as db_error:
            logging.error(f"Error updating status for client {client_id} in the database: {db_error}", exc_info=True)
        except Exception as e:
            logging.error(f"Error during final cleanup for client {client_id}: {e}", exc_info=True)
        finally:
            db.close()  # Ensure database session is always closed
            logging.info(f"Database session closed for client {client_id}.")


# Scheduler setup
scheduler = BackgroundScheduler()

@scheduler.scheduled_job(CronTrigger(minute="*/5"))
def scheduled_aggregate_weights():
    """
    Scheduled task to aggregate weights every minute.
    """
    logging.info("Scheduled task: Starting weight aggregation process.")
    try:
        asyncio.run(aggregate_weights())
    except Exception as e:
        logging.error(f"Error during scheduled weight aggregation: {e}")

scheduler.start()

# if __name__ == "__main__":
#     import uvicorn
#     logging.info("Starting Server...")
#     uvicorn.run(app, host="localhost", port=8000)