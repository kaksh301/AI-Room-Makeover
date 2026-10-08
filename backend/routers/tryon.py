from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

import os
import traceback
import base64
from io import BytesIO

from PIL import Image
from huggingface_hub import InferenceClient


# Load environment variables
load_dotenv()


# ============================================================
# HUGGING FACE CONFIGURATION
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("Missing HF_TOKEN in .env")


hf_client = InferenceClient(
    provider="fal-ai",
    api_key=HF_TOKEN,
)


router = APIRouter()


# ============================================================
# TRY-ON / ROOM REDESIGN
# ============================================================

@router.post("/try-on")
async def try_on(
    place_image: UploadFile = File(...),
    design_type: str = Form(...),
    room_type: str = Form(...),
    style: str = Form(...),
    background_color: str = Form(...),
    foreground_color: str = Form(...),
    instructions: str = Form(""),
):

    try:

        # ====================================================
        # 1. VALIDATE IMAGE
        # ====================================================

        MAX_IMAGE_SIZE_MB = 10

        ALLOWED_MIME_TYPES = {
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/heic",
            "image/heif",
        }

        if place_image.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type for place_image: {place_image.content_type}"
            )


        # ====================================================
        # 2. READ IMAGE
        # ====================================================

        place_bytes = await place_image.read()

        size_in_mb_for_place_image = len(place_bytes) / (1024 * 1024)

        if size_in_mb_for_place_image > MAX_IMAGE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail="Image exceeds 10MB size limit for place_image"
            )


        # ====================================================
        # 3. CREATE IMAGE EDITING PROMPT
        # ====================================================

        prompt = f"""
You are a professional interior and exterior designer.

Redesign the uploaded {room_type} according to the user's requirements.

USER REQUIREMENTS:

Design Type:
{design_type}

Room Type:
{room_type}

Style:
{style}

Background Color:
{background_color}

Foreground Color:
{foreground_color}

Additional Instructions:
{instructions}


IMPORTANT DESIGN RULES:

1. Keep the original room architecture unchanged.

2. Do NOT change:
   - walls
   - ceiling
   - floor structure
   - windows
   - doors
   - room dimensions
   - camera perspective

3. Only redesign the interior appearance.

4. Change or improve:
   - furniture
   - decoration
   - lighting
   - wall colors
   - curtains
   - beds
   - tables
   - wardrobes
   - carpets
   - plants
   - artwork
   - materials
   - accessories

5. Apply the requested {style} style.

6. Use the requested background and foreground colors
   naturally in the interior design.

7. Make the result highly photorealistic.

8. Make the furniture and decoration realistic and practical.

9. Preserve the original room layout as much as possible.

10. The final result should look like a professionally
    designed real room photographed with a high-quality camera.

Do not create a completely different room.

Transform the existing room.
"""


        print("\n================ PROMPT ================\n")
        print(prompt)
        print("\n========================================\n")


        # ====================================================
        # 4. SEND IMAGE TO HUGGING FACE
        # ====================================================

        print("Sending image to Hugging Face...")
        print("Model: Qwen/Qwen-Image-Edit")


        edited_image = hf_client.image_to_image(
            place_bytes,
            prompt=prompt,
            model="black-forest-labs/FLUX.2-klein-9B",
        )


        print("Image received from Hugging Face.")


        # ====================================================
        # 5. CONVERT GENERATED IMAGE TO BASE64
        # ====================================================

        image_buffer = BytesIO()

        edited_image.save(
            image_buffer,
            format="PNG"
        )

        image_bytes = image_buffer.getvalue()

        image_base64 = base64.b64encode(
            image_bytes
        ).decode("utf-8")


        image_url = (
            f"data:image/png;base64,{image_base64}"
        )


        # ====================================================
        # 6. TEXT RESPONSE
        # ====================================================

        text_response = (
            f"The {room_type} has been redesigned in a "
            f"{style} style while preserving the original "
            f"room structure and layout. The design uses "
            f"{background_color} as the background color and "
            f"{foreground_color} as the foreground color, "
            f"with realistic furniture, decoration and lighting."
        )


        # ====================================================
        # 7. RETURN RESPONSE TO FRONTEND
        # ====================================================

        return JSONResponse(
            content={
                "image": image_url,
                "text": text_response,
            }
        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except HTTPException:
        raise

    except Exception as e:

        print(
            f"Error in /api/try-on endpoint: {e}"
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )