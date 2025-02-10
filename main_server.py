import os
import logging
import numpy as np
import time
from typing import List, Optional, Dict, Tuple
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Response
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
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Tuple
import numpy as np
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

import os
from sqlalchemy import create_engine, Column, String, DateTime, Table, Integer, ForeignKey, select
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DisconnectionError
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base
from datetime import datetime

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")  # Format: postgresql://user:password@host:port/database
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is missing")
global_vars_runtime = {
    'last_aggregation_timestamp': 0,
    'latest_version': 0,
    'last_checked_timestamp': 0
}

# SQLAlchemy setup
engine = create_engine(
    DATABASE_URL,
    # pool_size=5,  # Maintain a pool of connections
    # max_overflow=10,  # Allow extra connections when needed
    # pool_recycle=600,  # Recycle connections every 30 min to prevent timeout
    pool_pre_ping=True,  # Ping connections before using them
)
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
class GlobalAggregation(Base):
    __tablename__ = "global_aggregation"
    
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
    except Exception as e:
        logging.error(f"Database error: {str(e)}")
        db.rollback()
        raise  # Re-raise the exception after rollback
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
        for attempt in range(3):
            try:
                if client_id in self.active_connections:
                    del self.active_connections[client_id]
                    logging.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
                break
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} - Error disconnecting client {client_id}: {e}")
                if attempt < 2:
                    time.sleep(2)

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
    last_aggregation_timestamp: int
) -> Optional[Tuple[List[Tuple[str, np.ndarray]], List[int], List[float], int]]:
    """
    Loads new weights from the Azure Blob Storage and extracts client IDs.

    Returns:
        - List of tuples (client_id, weights)
        - List of num_examples
        - List of loss values
        - Updated last_aggregation_timestamp
    """
    try:
        # Regex pattern to extract client_id and timestamp
        pattern = re.compile(r"client([0-9a-fA-F\-]+)_v\d+_(\d{8}_\d{6})\.keras")
        container_client = blob_service_client.get_container_client(container_name)

        weights_list = []
        num_examples_list = []
        loss_list = []
        new_last_aggregation_timestamp = last_aggregation_timestamp

        blobs = list(container_client.list_blobs())

        for blob in blobs:
            match = pattern.match(blob.name)
            if match:
                client_id = match.group(1)  # Extract client ID
                timestamp_str = match.group(2)
                timestamp_int = int(timestamp_str.replace("_", ""))

                if timestamp_int > last_aggregation_timestamp:
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
                        num_examples = int(blob_metadata.get('num_examples', 0))
                        loss = float(blob_metadata.get('loss', 0.0))
                        if num_examples == 0:
                            continue  # Skip blobs with no valid 'num_examples' metadata
                        num_examples_list.append(num_examples)
                        loss_list.append(loss)

                    # Clean up temporary file
                    os.unlink(temp_path)

                    # Store (client_id, weights)
                    weights_list.append((client_id, weights))
                    new_last_aggregation_timestamp = max(new_last_aggregation_timestamp, timestamp_int)

        if not weights_list:
            logging.info(f"No new weights found since {last_aggregation_timestamp}.")
            return None, [], [], last_aggregation_timestamp

        logging.info(f"Loaded weights from {len(weights_list)} files.")
        return weights_list, num_examples_list, loss_list, new_last_aggregation_timestamp

    except Exception as e:
        logging.error(f"Error loading weights: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return None, [], [], last_aggregation_timestamp
    
# Load the last processed timestamp from the database
def load_last_aggregation_timestamp(db: Session) -> int:
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            # Attempt to query the database
            timestamp = db.query(GlobalAggregation).filter_by(key="last_aggregation_timestamp").first()
            return int(timestamp.value) if timestamp else 0
        except OperationalError as db_error:
            logging.error(f"Attempt {attempt + 1} - Database error: {db_error}", exc_info=True)
            db.rollback()  # Rollback any pending transaction
            if attempt < retry_attempts - 1:
                time.sleep(2)  # Wait before retrying
            else:
                raise

# Save the last processed timestamp to the database
def save_last_aggregation_timestamp(db: Session, new_timestamp):
    try:
        # Fetch the existing timestamp record
        timestamp_record = db.query(GlobalAggregation).filter_by(key="last_aggregation_timestamp").first()
        
        if timestamp_record:
            # Update the existing record
            timestamp_record.value = new_timestamp
        else:
            # Insert a new record if not found
            new_record = GlobalAggregation(key="last_aggregation_timestamp", value=new_timestamp)
            db.add(new_record)
        
        db.commit()  # Commit changes
    except Exception as e:
        logging.error(f"Error saving last aggregation timestamp: {e}", exc_info=True)
        raise  # Re-raise exception for handling at a higher level


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

async def aggregate_weights_core(db: Session):
    try:
        global_vars_runtime['last_checked_timestamp'] = datetime.now().strftime("%Y%m%d%H%M%S")
        # Load last processed timestamp from the database
        last_aggregation_timestamp = load_last_aggregation_timestamp(db)
        global_vars_runtime['last_aggregation_timestamp'] = last_aggregation_timestamp or 0
        logging.info(f"Loaded last processed timestamp: {global_vars_runtime['last_aggregation_timestamp']}")

        model = get_model_architecture()
        if not model:
            logging.critical("Failed to load model architecture")
            raise HTTPException(status_code=500, detail="Failed to load model architecture")

        weights_list_with_ids, num_examples_list, loss_list, new_timestamp = load_weights_from_blob(
            blob_service_client_client, 
            CLIENT_CONTAINER_NAME, 
            model, 
            global_vars_runtime['last_aggregation_timestamp']
        )

        if not weights_list_with_ids:
            logging.info("No new weights found in the blob")
            return {"status": "no_update", "message": "No new weights found", "num_clients": 0}
        if len(weights_list_with_ids) < 2:
            logging.info("Insufficient weights for aggregation (only 1 weight file found)")
            return {"status": "no_update", "message": "Only 1 weight file found", "num_clients": 1}
        if not num_examples_list:
            logging.error("Example counts are missing, cannot perform aggregation")
            return {"status": "error", "message": "Example counts missing for aggregation"}

        # Synchronize latest version from the database
        latest_model = db.query(GlobalModel).order_by(GlobalModel.version.desc()).first()
        global_vars_runtime['latest_version'] = latest_model.version if latest_model else 0
        logging.info(f"Latest model version loaded: {global_vars_runtime['latest_version']}")

        # Increment version
        global_vars_runtime['latest_version'] += 1
        if db.query(GlobalModel).filter_by(version=global_vars_runtime['latest_version']).first():
            logging.error(f"Duplicate model version detected: {global_vars_runtime['latest_version']}")
            raise HTTPException(status_code=409, detail=f"Model with version {global_vars_runtime['latest_version']} already exists")

        filename = get_versioned_filename(global_vars_runtime['latest_version'])
        logging.info(f"Preparing to save aggregated weights as: {filename}")

        weights_list = [weights for _, weights in weights_list_with_ids]

        logging.info(f"Aggregating weights from {len(weights_list)} clients")
        avg_weights = federated_weighted_averaging(weights_list, num_examples_list, loss_list)
        logging.info("Aggregation completed successfully.")

        if not avg_weights or not save_weights_to_blob(avg_weights, filename, model):
            logging.critical("Failed to save aggregated weights to blob")
            raise HTTPException(status_code=500, detail="Failed to save aggregated weights")

        # Save the new timestamp to the database
        save_last_aggregation_timestamp(db, new_timestamp)  # Make sure to pass db here
        logging.info(f"New timestamp saved to the database: {new_timestamp}")

        # Update the database with model and client information
        contributing_client_ids = [id for id, _ in weights_list_with_ids]
        new_model = GlobalModel(
            version=global_vars_runtime['latest_version'],
            num_clients_contributed=len(weights_list),
            client_ids=",".join(contributing_client_ids)
        )
        db.add(new_model)

        db.query(Client).filter(Client.client_id.in_(contributing_client_ids)).update(
            {"contribution_count": Client.contribution_count + 1},
            synchronize_session=False
        )
        db.commit()
        logging.info(f"Model version {global_vars_runtime['latest_version']} saved and database updated")

        # Notify clients of new model
        await manager.broadcast_model_update(f"NEW_MODEL:{filename}")
        logging.info(f"Clients notified of new model: {filename}")

        return {
            "status": "success",
            "message": f"Aggregated weights saved as {filename}",
            "num_clients": len(weights_list)
        }
    except SQLAlchemyError as db_error:
        logging.error(f"Database error during aggregation: {db_error}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="Database error occurred") from db_error
    except HTTPException as http_exc:
        logging.error(f"HTTP Exception: {http_exc.detail}")
        db.rollback()
        raise
    except Exception as e:
        logging.exception("Unexpected error during aggregation")
        db.rollback()
        raise HTTPException(status_code=500, detail="An unexpected error occurred") from e


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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "HELLO, WORLD. Welcome to the AdaptFL Server!"}

@app.get("/health", response_class=JSONResponse)
async def health_check():
    """
    Health check endpoint to verify service availability.
    """
    return {"status": "healthy"}

@app.head("/health")
async def health_check_monitor():
    """
    Health check endpoint to verify service availability.
    """
    return Response(status_code=200)

@app.get("/get_data")
async def get_all_data(db: Session = Depends(get_db)):
    try:
        # Query all data from the 'clients' table
        clients = db.execute(select(Client)).scalars().all()

        # Query all data from the 'global_models' table
        global_models = db.execute(select(GlobalModel)).scalars().all()

        # Query all data from the 'global_vars' table
        global_vars_table = db.execute(select(GlobalAggregation)).scalars().all()

        # Return all the data as a dictionary
        return {
            "clients": clients,
            "global_models": global_models,
            "global_aggregation": global_vars_table,
            "last_checked_timestamp": global_vars_runtime['last_checked_timestamp']
        }
    except Exception as e:
        logging.error(f"Error in /get_data endpoint: {e}", exc_info=True)
        return {"error": "Failed to fetch data. Please try again later."}

# Modified registration endpoint
@app.post("/register")
async def register(
    csn: str = Body(..., embed=True),
    admin_api_key: str = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    try:
        # Verify admin API key
        verify_admin(admin_api_key)
        
        # Check if the client is already registered
        existing_client = db.query(Client).filter(Client.csn == csn).first()
        if existing_client:
            raise HTTPException(status_code=400, detail="Client already registered")
        
        # Generate unique client ID and API key
        client_id = str(uuid.uuid4())
        api_key = str(uuid.uuid4())
        
        # Create a new client record
        new_client = Client(
            csn=csn,
            client_id=client_id,
            api_key=api_key
        )
        db.add(new_client)
        db.commit()
        
        # Return success response
        return {
            "status": "success",
            "message": "Client registered successfully",
            "data": {"client_id": client_id, "api_key": api_key}
        }
    except HTTPException as http_exc:
        raise http_exc  # Re-raise HTTP exceptions to be handled by FastAPI
    except Exception as e:
        # Log unexpected errors and return a generic error response
        logging.error(f"Error during client registration: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An error occurred while processing the registration"
        )

@app.get("/aggregate-weights")
async def aggregate_weights(db: Session = Depends(get_db)):
    try:
        return await aggregate_weights_core(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str, db: Session = Depends(get_db)):
    retry_attempts = 3
    try:
        # Check if the client exists in the database
        client = db.query(Client).filter(Client.client_id == client_id).first()
        if not client:
            logging.warning(f"Client {client_id} not found in database. Closing WebSocket.")
            await websocket.close(code=1008, reason="Unauthorized")
            return
        logging.info(f"Client {client_id} found in DB: {client}")

        # Add the client to the WebSocket manager and update status
        await manager.connect(client_id, websocket)

        # Retry updating the status to "Active" if an error occurs
        for attempt in range(retry_attempts):
            try:
                client.status = "Active"
                db.commit()
                logging.info(f"Client {client_id} connected successfully, status updated to 'Active'.")
                break
            except SQLAlchemyError as db_error:
                db.rollback()  # Added explicit rollback
                logging.error(f"Attempt {attempt + 1} - Failed to update 'Active' status for {client_id}: {db_error}")
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(2)  # Changed to asyncio.sleep for async context
                else:
                    raise

        # Inform the client about their updated status
        await websocket.send_text(f"Your status is now: {client.status}")

        # Get the latest model version and send to the client
        latest_model = db.query(GlobalModel).order_by(GlobalModel.version.desc()).first()
        global_vars_runtime['latest_version'] = latest_model.version if latest_model else 0
        latest_model_version = f"g{global_vars_runtime['latest_version']}.keras"
        await websocket.send_text(f"LATEST_MODEL:{latest_model_version}")

        while True:
            try:
                # Handle messages from the client
                data = await websocket.receive_text()
                if not data:
                    break

                # Update and notify client if status changes
                for attempt in range(retry_attempts):
                    try:
                        # Refresh the session to avoid stale data
                        db.refresh(client)
                        if client.status != "Active":
                            client.status = "Active"
                            db.commit()
                            await websocket.send_text(f"Your updated status is: {client.status}")
                        break
                    except SQLAlchemyError as db_error:
                        db.rollback()  # Added explicit rollback
                        logging.error(f"Attempt {attempt + 1} - Database error for client {client_id}: {db_error}")
                        if attempt < retry_attempts - 1:
                            await asyncio.sleep(2)
                        else:
                            raise

            except WebSocketDisconnect:
                logging.info(f"Client {client_id} disconnected gracefully.")
                break
            except Exception as e:
                logging.error(f"Error handling message from client {client_id}: {e}")
                await websocket.send_text("An error occurred. Please try again later.")
                break

    except SQLAlchemyError as db_error:
        logging.error(f"Database error for client {client_id}: {db_error}", exc_info=True)
        db.rollback()  # Added explicit rollback
        await websocket.close(code=1002, reason="Database error.")
    except WebSocketDisconnect:
        logging.info(f"Client {client_id} disconnected unexpectedly.")
    except Exception as e:
        logging.error(f"Unexpected error for client {client_id}: {e}", exc_info=True)
        await websocket.close(code=1000, reason="Internal server error.")
    finally:
        # Cleanup: Disconnect the client and update the database
        for attempt in range(retry_attempts):
            try:
                await manager.disconnect(client_id)
                break
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} - Failed to disconnect client {client_id}: {e}")
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(2)

        for attempt in range(retry_attempts):
            try:
                # Refresh the client object to avoid stale data
                if client:  # Ensure client is valid before updating
                    db.refresh(client)  # Refresh to prevent stale data
                    client.status = "Inactive"
                    db.commit()
                    logging.info(f"Client {client_id} is now inactive. DB updated successfully.")
                else:
                    logging.warning(f"Skipping status update: Client {client_id} does not exist.")

                break
            except SQLAlchemyError as db_error:
                db.rollback()  # Added explicit rollback
                logging.error(f"Attempt {attempt + 1} - Failed to update 'Inactive' status for {client_id}: {db_error}")
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(2)
                else:
                    raise
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} - Unexpected error during cleanup for {client_id}: {e}")
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(2)

        logging.info(f"Cleanup completed for client {client_id}.")


# Scheduler setup
scheduler = BackgroundScheduler()

@scheduler.scheduled_job(CronTrigger(hours="*/6"))
def scheduled_aggregate_weights():
    """
    Scheduled task to aggregate weights every minute.
    """
    logging.info("Scheduled task: Starting weight aggregation process.")
    db = SessionLocal()
    try:
        asyncio.run(aggregate_weights_core(db))
    except Exception as e:
        logging.error(f"Error during scheduled weight aggregation: {e}")
    finally:
        db.close()
scheduler.start()

# if __name__ == "__main__":
#     import uvicorn
#     logging.info("Starting Server...")
#     uvicorn.run(app, host="localhost", port=8000)