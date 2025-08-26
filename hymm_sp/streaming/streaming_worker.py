import os
import sys
import torch
import torch.distributed as dist
from loguru import logger
import time
import signal
import json
import traceback
from PIL import Image
import cv2
import torchvision.transforms as transforms
from hymm_sp.sample_inference import HunyuanVideoSampler
from hymm_sp.data_kits.data_tools import save_videos_grid
from hymm_sp.modules.parallel_states import (
    initialize_distributed,
    nccl_info,
)

# ========== Configuration ==========
TRIGGER_FILE = "./streaming_results/trigger.txt"
RESULT_DIR = "./streaming_results"
INPUT_IMAGES_DIR = os.path.join(RESULT_DIR, "input")
SHUTDOWN_FILE = "./streaming_results/shutdown_worker.signal"
CKPT_PATH = os.getenv("CKPT_PATH")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(INPUT_IMAGES_DIR, exist_ok=True)

# Global variables
sampler = None
args = None
shutdown_flag = False
active_streams = {}  # Track active streaming sessions

class CropResize:
    """Crop-style resize transformation"""
    def __init__(self, size=(352, 608)):  # Default to launch-distill.sh size
        self.target_h, self.target_w = size 
        
    def __call__(self, img):
        w, h = img.size
        scale = max(
            self.target_w / w,
            self.target_h / h
        )
        new_size = (int(h * scale), int(w * scale))
        resize_transform = transforms.Resize(new_size, 
                                           interpolation=transforms.InterpolationMode.BILINEAR)
        resized_img = resize_transform(img)
        crop_transform = transforms.CenterCrop((self.target_h, self.target_w))
        return crop_transform(resized_img)

class StreamingSession:
    """Track state for a streaming session"""
    def __init__(self, stream_id, params):
        self.stream_id = stream_id
        self.params = params
        self.last_frame = None
        self.frame_count = 0

def signal_handler(sig, frame):
    global shutdown_flag
    rank = dist.get_rank() if dist.is_initialized() else 0
    logger.info(f"Rank {rank} received shutdown signal.")
    shutdown_flag = True

def load_model():
    """Load model with distributed setup"""
    global sampler, args
    from hymm_sp.config import parse_args
    
    rank = torch.distributed.get_rank()
    if rank == 0:
        print("="*60)
        print("🚀 STARTING MODEL LOADING PROCESS")
        print("="*60)
    
    args = parse_args()
    
    # Set default parameters from launch-distill.sh
    args.ckpt = CKPT_PATH
    args.add_neg_prompt = "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border."
    args.input_prompt = "Realistic, High-quality."
    args.video_size = [352, 608]
    args.cfg_scale = 1.0
    args.image_start = True
    args.action_list = ['w']
    args.action_speed_list = [0.2]
    args.seed = 250161
    args.infer_steps = 8
    args.use_fp8 = True
    args.flow_shift_eval_video = 5.0
    args.sample_n_frames = 9
    args.save_path = "./streaming_results"
    
    rank = torch.distributed.get_rank()
    if rank == 0:
        print("🔧 Loading streaming model...")
        print(f"🔧 Checkpoint: {args.ckpt}")
        print(f"🔧 Video size: {args.video_size}")
        print(f"🔧 Inference steps: {args.infer_steps}")
        logger.info("🔧 Loading streaming model...")
        logger.info(f"🔧 Checkpoint: {args.ckpt}")
        logger.info(f"🔧 Video size: {args.video_size}")
        logger.info(f"🔧 Inference steps: {args.infer_steps}")
        
    try:
        device = torch.device("cuda")
        if nccl_info.sp_size > 1:
            device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        sampler = HunyuanVideoSampler.from_pretrained(args.ckpt, args=args, device=device)
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error(traceback.format_exc())
        raise e
    
    if rank == 0:
        print("="*60)
        print("✅ MODEL LOADED SUCCESSFULLY! Ready for streaming.")
        print("="*60)
        logger.info("✅ Model loaded. Ready for streaming.")

def process_streaming_request(trigger_data, rank, device):
    """Process a streaming video generation request"""
    global sampler, args, active_streams
    
    stream_id = trigger_data.get("stream_id")
    custom_params = trigger_data.get("custom_params", {})
    
    if rank == 0:
        logger.info(f"🎬 Processing stream request: {stream_id}")
        logger.info(f"📊 Custom params: {json.dumps(custom_params, indent=2)}")
    
    # Create a copy of args to avoid interference between requests
    import copy
    request_args = copy.deepcopy(args)
        
    # Get or create streaming session
    if stream_id not in active_streams:
        active_streams[stream_id] = StreamingSession(stream_id, custom_params)
        if rank == 0:
            logger.info(f"🆕 Created new session for {stream_id}")
    else:
        if rank == 0:
            logger.info(f"♻️ Reusing existing session for {stream_id}")
    
    session = active_streams[stream_id]
    
    # Update request_args with custom parameters - ALWAYS update to get latest action
    if rank == 0:
        logger.info(f"📋 Before update - request_args.action_list: {request_args.action_list if hasattr(request_args, 'action_list') else 'NOT SET'}")
    
    if custom_params:
        # Video generation parameters
        request_args.image_path = custom_params.get("image_path", request_args.image_path if hasattr(request_args, 'image_path') else None)
        request_args.input_prompt = custom_params.get("prompt", request_args.input_prompt)
        request_args.add_neg_prompt = custom_params.get("add_neg_prompt", request_args.add_neg_prompt)
        request_args.video_size = custom_params.get("video_size", request_args.video_size)
        request_args.cfg_scale = custom_params.get("cfg_scale", request_args.cfg_scale)
        request_args.seed = custom_params.get("seed", request_args.seed)
        request_args.infer_steps = custom_params.get("infer_steps", request_args.infer_steps)
        request_args.use_fp8 = custom_params.get("use_fp8", request_args.use_fp8)
        request_args.flow_shift_eval_video = custom_params.get("flow_shift_eval_video", request_args.flow_shift_eval_video)
        request_args.sample_n_frames = custom_params.get("sample_n_frames", request_args.sample_n_frames)
        
        # Action control - ensure proper format
        action_list = custom_params.get("action_list", ['w'])
        action_speed_list = custom_params.get("action_speed_list", [0.2])
        
        # Log what we're getting
        if rank == 0:
            logger.info(f"🎯 NEW ACTION REQUEST:")
            logger.info(f"   - Raw action_list from params: {action_list}, type: {type(action_list)}")
            logger.info(f"   - Raw action_speed_list from params: {action_speed_list}, type: {type(action_speed_list)}")
            logger.info(f"   - Previous request_args.action_list: {request_args.action_list if hasattr(request_args, 'action_list') else 'None'}")
        
        # ALWAYS update with new action from request
        if isinstance(action_list, list) and len(action_list) > 0:
            request_args.action_list = action_list
            if rank == 0:
                logger.info(f"   ✅ Updated request_args.action_list to: {request_args.action_list}")
        else:
            request_args.action_list = ['w']
            if rank == 0:
                logger.info(f"   ⚠️ Invalid action_list, defaulting to: {request_args.action_list}")
            
        if isinstance(action_speed_list, list) and len(action_speed_list) > 0:
            request_args.action_speed_list = action_speed_list
        else:
            request_args.action_speed_list = [0.2]
            
        # Ensure lists have same length
        while len(request_args.action_speed_list) < len(request_args.action_list):
            request_args.action_speed_list.append(0.2)
            
        if rank == 0:
            logger.info(f"   ✅ Final action_list: {request_args.action_list}")
            logger.info(f"   ✅ Final action_speed_list: {request_args.action_speed_list}")
    
    try:
        # Initialize latents
        ref_latents = None
        last_latents = None
        
        # Load and prepare input image
        if request_args.image_path and os.path.exists(request_args.image_path):
            ref_image = Image.open(request_args.image_path).convert("RGB")
            
            # Use the frame from previous generation if continuing
            if session.last_frame is not None and session.frame_count > 0:
                ref_image = session.last_frame
            
            # Prepare transform
            ref_image_transform = transforms.Compose([
                CropResize(tuple(request_args.video_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            
            ref_image_tensor = ref_image_transform(ref_image).unsqueeze(0).unsqueeze(2)
            if device.type == "cuda":
                ref_image_tensor = ref_image_tensor.to(device=device, dtype=torch.float16)
            
            # Encode to latents for first frame
            if session.frame_count == 0:
                with torch.no_grad():
                    ref_latents = sampler.vae.encode(ref_image_tensor).latent_dist.sample()
                    ref_latents = ref_latents * sampler.vae.config.scaling_factor
                    # For image_start, last_latents is initialized to ref_latents
                    last_latents = ref_latents.clone()
                    if rank == 0:
                        logger.info(f"📊 Encoded ref_latents shape: {ref_latents.shape}")
                        logger.info(f"📊 Initialized last_latents shape: {last_latents.shape}")
        else:
            # For image_start mode, we need at least a blank image
            if session.frame_count == 0 and request_args.image_start:
                # Create a blank image tensor - shape: (batch, channels, frames, height, width)
                # request_args.video_size is [height, width] = [352, 608]
                ref_image_tensor = torch.zeros(1, 3, 1, request_args.video_size[0], request_args.video_size[1])
                if device.type == "cuda":
                    ref_image_tensor = ref_image_tensor.to(device=device, dtype=torch.float16)
                
                with torch.no_grad():
                    ref_latents = sampler.vae.encode(ref_image_tensor).latent_dist.sample()
                    ref_latents = ref_latents * sampler.vae.config.scaling_factor
                    # For image_start, last_latents is initialized to ref_latents
                    last_latents = ref_latents.clone()
                    if rank == 0:
                        logger.info(f"📊 Created blank ref_latents shape: {ref_latents.shape}")
                        logger.info(f"📊 Initialized blank last_latents shape: {last_latents.shape}")
        
        # Use stored latents for continuation
        if session.frame_count > 0 and hasattr(session, 'last_latents'):
            last_latents = session.last_latents
            ref_latents = session.ref_latents if hasattr(session, 'ref_latents') else None
            if rank == 0:
                if last_latents is not None:
                    logger.info(f"📊 Using stored last_latents shape: {last_latents.shape}")
                if ref_latents is not None:
                    logger.info(f"📊 Using stored ref_latents shape: {ref_latents.shape}")
            
        # Generate video chunk
        if rank == 0:
            logger.info(f"🎬 Generating video chunk for stream {stream_id}")
            logger.info(f"🎮 Action: {request_args.action_list}, Speed: {request_args.action_speed_list}")
            logger.info(f"📐 Video size: {request_args.video_size[0]}x{request_args.video_size[1]}")
            logger.info(f"🎯 Frames: {request_args.sample_n_frames}, Steps: {request_args.infer_steps}")
            logger.info(f"🌱 Seed: {request_args.seed + session.frame_count}")
        
        # Call sampler with verbose output
        if rank == 0:
            logger.info("⚡ Starting inference...")
            logger.info(f"📊 ref_latents: {ref_latents.shape if ref_latents is not None else None}")
            logger.info(f"📊 last_latents: {last_latents.shape if last_latents is not None else None}")
        
        # Get the first action from the list (for streaming, we process one action at a time)
        if rank == 0:
            logger.info(f"🔍 EXTRACTING ACTION FROM request_args.action_list: {request_args.action_list}")
            
        action_id = request_args.action_list[0] if request_args.action_list else 'w'
        action_speed = request_args.action_speed_list[0] if request_args.action_speed_list else 0.2
        
        if rank == 0:
            logger.info(f"🎮 EXTRACTED action_id: '{action_id}' (type: {type(action_id)})")
            logger.info(f"🎮 EXTRACTED action_speed: {action_speed}")
            
            # Double-check the action is correct
            if action_id == 'd':
                logger.warning(f"⚠️ ACTION IS 'd' (RIGHT) - Checking if this is intended")
            elif action_id == 'w':
                logger.info(f"✅ ACTION IS 'w' (FORWARD)")
            elif action_id == 'a':
                logger.info(f"✅ ACTION IS 'a' (LEFT)")
            elif action_id == 's':
                logger.info(f"✅ ACTION IS 's' (BACKWARD)")
            else:
                logger.error(f"❌ UNKNOWN ACTION: '{action_id}'")
        
        is_image_mode = session.frame_count == 0 and request_args.image_start
        
        if rank == 0:
            logger.info(f"🎥 is_image: {is_image_mode}")
            logger.info(f"📹 Frame count: {session.frame_count}, video_length: {request_args.sample_n_frames}")
        
        # Ensure all ranks are synchronized before inference
        dist.barrier()
            
        samples = sampler.predict(
            prompt=request_args.input_prompt,
            size=request_args.video_size,  # Pass as tuple (height, width)
            video_length=request_args.sample_n_frames,
            seed=request_args.seed + session.frame_count,  # Vary seed for each chunk
            negative_prompt=request_args.add_neg_prompt,
            infer_steps=request_args.infer_steps,
            guidance_scale=request_args.cfg_scale,
            num_videos_per_prompt=1,
            flow_shift=request_args.flow_shift_eval_video,
            ref_latents=ref_latents,  # Pass encoded latents
            last_latents=last_latents,  # Previous frame latents  
            action_id=action_id,  # Single action, not list
            action_speed=action_speed,  # Single speed value, not list
            is_image=is_image_mode,  # First chunk with image_start
            return_latents=True  # Need latents for next iteration
        )
        
        if rank == 0:
            logger.info("✅ Inference completed!")
            
            # Check if samples is a dict with latents
            if isinstance(samples, dict):
                # Get video samples - it's a list, take first element
                video_samples_list = samples.get("samples", [])
                if video_samples_list and len(video_samples_list) > 0:
                    video_samples = video_samples_list[0]  # Get first element like sample_batch.py
                    logger.info(f"📐 Video samples shape: {video_samples.shape}")
                else:
                    logger.error("No samples in output!")
                    video_samples = None
                    
                # Store latents for next iteration
                if "last_latents" in samples:
                    session.last_latents = samples["last_latents"]
                    logger.info(f"📊 Stored last_latents shape: {session.last_latents.shape}")
                if "ref_latents" in samples:
                    session.ref_latents = samples["ref_latents"]
                    logger.info(f"📊 Stored ref_latents shape: {session.ref_latents.shape}")
            else:
                video_samples = samples
            
            # Save video chunk
            if video_samples is not None:
                timestamp = int(time.time() * 1000)
                video_path = os.path.join(RESULT_DIR, f"stream_{stream_id}_{timestamp}.mp4")
                
                logger.info(f"💾 Saving video to: {video_path}")
                save_videos_grid(
                    video_samples,
                    video_path,
                    n_rows=1,
                    fps=30
                )
                logger.info(f"✅ Video saved successfully!")
            else:
                logger.error("Cannot save video - no samples generated")
                video_path = None
            
            # Extract last frame for next iteration
            if video_path:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        session.last_frame = Image.fromarray(frame_rgb)
                    cap.release()
            
            session.frame_count += request_args.sample_n_frames
            
            # Write result
            result = {
                "stream_id": stream_id,
                "video_path": video_path,
                "frame_count": session.frame_count,
                "status": "success"
            }
            
            result_file = os.path.join(RESULT_DIR, f"result_{stream_id}.json")
            logger.info(f"📝 Writing result to: {result_file}")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"✅ Result written successfully!")
                
            logger.info(f"🎉 Completed chunk generation for stream {stream_id}")
        else:
            # Non-rank-0 processes also need to update session state and latents
            session.frame_count += request_args.sample_n_frames
            if isinstance(samples, dict):
                if "last_latents" in samples:
                    session.last_latents = samples["last_latents"]
                if "ref_latents" in samples:
                    session.ref_latents = samples["ref_latents"]
            
    except Exception as e:
        if rank == 0:
            logger.error(f"Error in streaming generation: {e}")
            logger.error(traceback.format_exc())
            
            # Write error result
            result = {
                "stream_id": stream_id,
                "error": str(e),
                "status": "error"
            }
            
            result_file = os.path.join(RESULT_DIR, f"result_{stream_id}.json")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)

def worker_loop(rank, world_size, device):
    """Main worker loop for streaming"""
    global shutdown_flag, sampler, args
    
    logger.info(f"🔥 Streaming worker rank {rank} ready...")
    logger.info(f"📁 Monitoring trigger file: {TRIGGER_FILE}")
    logger.info(f"📁 Result directory: {RESULT_DIR}")
    
    loop_count = 0
    while not shutdown_flag:
        should_generate = False
        trigger_data = None
        
        if rank == 0:
            loop_count += 1
            if loop_count % 100 == 0:  # Log every 100 loops (~10 seconds)
                logger.debug(f"Worker alive - loop {loop_count}")
            
            # Check for trigger file
            if os.path.exists(TRIGGER_FILE):
                try:
                    with open(TRIGGER_FILE, 'r') as f:
                        trigger_data = json.load(f)
                    
                    if isinstance(trigger_data, dict) and "stream_id" in trigger_data:
                        should_generate = True
                        logger.info(f"✅ Found streaming trigger: {trigger_data['stream_id']}")
                        logger.info(f"📋 Trigger data: {json.dumps(trigger_data, indent=2)}")
                    
                except Exception as e:
                    logger.error(f"Error reading trigger: {e}")
                finally:
                    try:
                        os.remove(TRIGGER_FILE)
                    except:
                        pass
            
            # Check for shutdown
            if os.path.exists(SHUTDOWN_FILE):
                logger.info("Shutdown signal detected")
                shutdown_flag = True
                try:
                    os.remove(SHUTDOWN_FILE)
                except:
                    pass
        
        # Broadcast signals
        should_generate_tensor = torch.tensor([int(should_generate)], dtype=torch.long, device=device)
        dist.broadcast(should_generate_tensor, src=0)
        should_generate = bool(should_generate_tensor.item())
        
        shutdown_tensor = torch.tensor([int(shutdown_flag)], dtype=torch.long, device=device)
        dist.broadcast(shutdown_tensor, src=0)
        shutdown_flag = bool(shutdown_tensor.item())
        
        if should_generate:
            # Broadcast trigger data
            if rank == 0:
                trigger_str = json.dumps(trigger_data)
                trigger_bytes = trigger_str.encode('utf-8')
                trigger_tensor = torch.ByteTensor(list(trigger_bytes)).to(device)
                length_tensor = torch.tensor([len(trigger_bytes)], dtype=torch.long, device=device)
            else:
                length_tensor = torch.tensor([0], dtype=torch.long, device=device)
                trigger_tensor = torch.ByteTensor([0]).to(device)
            
            dist.broadcast(length_tensor, src=0)
            trigger_length = length_tensor.item()
            
            if trigger_length > 0:
                if rank != 0:
                    trigger_tensor = torch.ByteTensor(trigger_length).to(device)
                dist.broadcast(trigger_tensor, src=0)
                
                if rank != 0:
                    trigger_bytes = bytes(trigger_tensor.cpu().numpy().tolist())
                    trigger_str = trigger_bytes.decode('utf-8')
                    trigger_data = json.loads(trigger_str)
            
            # Process request
            process_streaming_request(trigger_data, rank, device)
        
        time.sleep(0.1)
    
    logger.info(f"Worker rank {rank} shutting down...")

def main():
    # Initialize distributed
    initialize_distributed(seed=42)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device(f"cuda:{rank}")
    
    # Register signal handler
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    if rank == 0:
        logger.info(f"🚀 Starting streaming worker with {world_size} GPUs...")
    
    # Load model
    load_model()
    
    # Enter main loop
    try:
        worker_loop(rank, world_size, device)
    except Exception as e:
        logger.error(f"Fatal error in worker rank {rank}: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()