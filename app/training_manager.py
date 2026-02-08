import asyncio
import threading
import queue
from typing import List, Optional
from fastapi import WebSocket

class TrainingManager:
    def __init__(self):
        self.is_training = False
        self.current_task: Optional[threading.Thread] = None
        self.log_queue = queue.Queue()
        self.active_websockets: List[WebSocket] = []
        self.stop_event = threading.Event()
        self.loop = None  # Reference to the main event loop

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_websockets.append(websocket)
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "is_training": self.is_training
        })

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_websockets:
            self.active_websockets.remove(websocket)

    async def broadcast_log(self, message: str):
        """Send a log message to all connected clients"""
        if not self.active_websockets:
            return
            
        payload = {
            "type": "log",
            "message": message
        }
        
        # We need to copy the list to avoid modification during iteration issues
        for ws in list(self.active_websockets):
            try:
                await ws.send_json(payload)
            except Exception as e:
                print(f"Error sending to websocket: {e}")
                self.disconnect(ws)

    async def broadcast_status(self, is_training: bool):
        """Send status update"""
        self.is_training = is_training
        payload = {
            "type": "status",
            "is_training": is_training
        }
        for ws in list(self.active_websockets):
            try:
                await ws.send_json(payload)
            except:
                self.disconnect(ws)

    def log_callback(self, message: str):
        """Callback to be used by the training script"""
        # Remove direct print if you want to silence console, 
        # or keep it to show logs in backend terminal too
        print(f"[Training] {message}")
        
        # Since this runs in a separate thread, we need to schedule 
        # the broadcast on the main event loop
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.broadcast_log(message), 
                self.loop
            )

    def start_training(self, training_function, config):
        if self.is_training:
            raise Exception("Training is already in progress")

        self.stop_event.clear()
        self.loop = asyncio.get_running_loop()
        
        # Create a wrapper to run the training function
        def target():
            try:
                # Notify start
                asyncio.run_coroutine_threadsafe(
                    self.broadcast_status(True), 
                    self.loop
                )
                
                # Run the actual training
                training_function(config, log_callback=self.log_callback)
                
                self.log_callback("Training completed successfully!")
            except Exception as e:
                self.log_callback(f"Training failed: {str(e)}")
                import traceback
                self.log_callback(traceback.format_exc())
            finally:
                # Notify end
                self.is_training = False
                if self.loop and self.loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self.broadcast_status(False), 
                        self.loop
                    )

        self.current_task = threading.Thread(target=target)
        self.current_task.start()
