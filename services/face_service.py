import os
import cv2
import asyncio
import pickle
import numpy as np
import unicodedata
import re
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import update, text
from sqlalchemy.future import select
from fastapi import HTTPException, Response, status
import json

from models.face import Face
from models.face_information import FaceInformation
from models.face_groupinfo import FaceGroupInfo
from schemas.FaceInfo import FaceInfo
import RegisterFace as rgf
from RegisterFace import face_angles, face_embeddings, captured_images
import video_stream as vs

async def reset_face_angles():
    rgf.face_angles = {key: False for key in face_angles}
    rgf.face_embeddings = {key: None for key in face_embeddings}
    rgf.captured_images = {key: None for key in captured_images}
    print("Reset face angles and captured images.", face_angles)

def register_face_async(stop_event):
    while not stop_event.is_set() and not all(face_angles.values()):
        rgf.register_faceByFrame(vs.latest_frame)
        print("Registering...")

def normalize_filename(name):
    name = unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode('utf-8')
    name = re.sub(r'\s+', '', name)
    name = re.sub(r'[^\w.-]', '', name)
    return name

async def save_face_info(face_info: FaceInfo, db: AsyncSession):
    if not face_info.email.endswith("@gmail.com"):
        return Response(
            content=json.dumps({"status": "error", "message": "Email must be in the format '@gmail.com'."}),
            status_code=status.HTTP_400_BAD_REQUEST,
            media_type="application/json"
        )

    if not face_info.firstName.isalpha() or not face_info.lastName.isalpha():
        return Response(
            json.dumps({"status": "error", "message": "Name must contain only alphabetic characters."}),
            status_code=status.HTTP_400_BAD_REQUEST,
            media_type="application/json"
        )

    if not isinstance(face_info.empId, str):
        return Response(
            json.dumps({"status": "error", "message": "Employee ID must be a string."}),
            status_code=status.HTTP_400_BAD_REQUEST,
            media_type="application/json"
        )

    if not isinstance(face_info.phone, str) or len(face_info.phone) != 10 or not face_info.phone.isdigit():
        return Response(
            json.dumps({"status": "error", "message": "Phone number must be a 10-digit string."}),
            status_code=status.HTTP_400_BAD_REQUEST,
            media_type="application/json"
        )

    # Kết hợp firstName và lastName thành name
    face_info.name = f"{face_info.firstName} {face_info.lastName}"

    result = await db.execute(select(FaceInformation.faceInfoId).order_by(FaceInformation.faceInfoId.desc()).limit(1))
    last_face = result.scalar()
    new_id = (last_face + 1) if last_face else 1

    group_result = await db.execute(select(FaceGroupInfo.id).where(FaceGroupInfo.groupName == face_info.groupName))
    group_id = group_result.scalar() or 0

    insert_sql = text(
        """
        CALL sp_insert_target(:id_, :empId, :firstName, :lastName, :groupId, :dob, 
                              :gender, :phone, :email, :avatar, @output);
        """
    )

    await db.execute(
        insert_sql,
        {
            "id_": new_id,
            "empId": face_info.empId,
            "firstName": face_info.firstName,
            "lastName": face_info.lastName,
            "groupId": group_id,
            "dob": face_info.dob,
            "gender": face_info.gender,
            "phone": face_info.phone,
            "email": face_info.email,
            "avatar": ""
        }
    )

    output_sql = text("SELECT @output AS refValue;")
    output_result = await db.execute(output_sql)
    ref_value = output_result.scalar()

    if ref_value == 1:
        if len(captured_images) < 4:
            return Response(
                content=json.dumps({"status": "error", "message": "Not enough face angles captured. Please complete all 5 angles before saving."}),
                status_code=status.HTTP_400_BAD_REQUEST,
                media_type="application/json"
            )
        
        normalized_last_name = normalize_filename(face_info.lastName)
        normalized_first_name = normalize_filename(face_info.firstName)
        normalized_group_name = normalize_filename(face_info.groupName)
        normalized_group_name_ClusterA = "ClusterA"
        normalized_dob = re.sub(r'\s+', '', face_info.dob)
        concatenated_name_dob = f"{normalized_first_name}{normalized_last_name}-{normalized_dob}"

        face_database_path = os.getenv('FACE_DATABASE_PATH')
        save_dir = f"{face_database_path}/{normalized_group_name_ClusterA}/{new_id}"
        os.makedirs(save_dir, exist_ok=True)
        save_dir_1 = f"{face_database_path}/{normalized_group_name}/{concatenated_name_dob}"
        os.makedirs(save_dir_1, exist_ok=True)

        image_paths = []
        for angle, img in captured_images.items():
            img_path = os.path.normpath(os.path.join(save_dir, f"{angle}.jpg"))
            img_path1 = os.path.normpath(os.path.join(save_dir_1, f"{angle}.jpg"))

            try:
                if cv2.imwrite(img_path, img) and cv2.imwrite(img_path1, img):
                    print(f"✅ Saved image: {img_path}")
                    image_paths.append(img_path1)
                    
                    if angle in face_embeddings and face_embeddings[angle] is not None:
                        embedding = face_embeddings[angle]
                        embedding_blob = embedding.tobytes()
                        if len(embedding_blob) > 2048:
                            embedding_blob = embedding_blob[:2048]
                        elif len(embedding_blob) < 2048:
                            embedding_blob = embedding_blob.ljust(2048, b'\0')

                        new_face = Face(
                            direction=angle,
                            faceInfoId=new_id,
                            embedding=embedding_blob
                        )
                        db.add(new_face)
                        print(f"✅ Added face record for angle: {angle}")
                else:
                    print(f"❌ Failed to save image: {img_path}")
            except cv2.error as e:
                print(f"❌ Error saving image {img_path}: {e}")

        avatar_paths = ",".join(image_paths)

        # Cập nhật cả avatar và name trong bảng FaceInformation
        await db.execute(
            update(FaceInformation)
            .where(FaceInformation.faceInfoId == new_id)
            .values(
                avatar=avatar_paths,
                name=face_info.name
            )
        )

        await db.commit()

        await reset_face_angles()

        return Response(
            content=json.dumps({
                "status": "success",
                "message": "Face data and images saved successfully",
                "data": {
                    "faceInfoId": new_id,
                    "empId": face_info.empId,
                    "firstName": face_info.firstName,
                    "lastName": face_info.lastName,
                    "name": face_info.name,
                    "groupId": group_id,
                    "groupName": face_info.groupName,
                    "dob": face_info.dob,
                    "gender": face_info.gender,
                    "phone": face_info.phone,
                    "email": face_info.email,
                    "avatar": avatar_paths
                }
            }),
            status_code=status.HTTP_201_CREATED,
            media_type="application/json"
        )

    else:
        return Response(
            json.dumps({"status": "error", "message": "Failed to save face data"}),
            status_code=status.HTTP_400_BAD_REQUEST,
            media_type="application/json"
        )

async def get_face_embedding(face_id: int, db: AsyncSession):
    result = await db.execute(
        select(Face.direction, Face.embedding)
        .where(Face.faceInfoId == face_id)
    )
    faces = result.all()

    if not faces:
        raise HTTPException(status_code=404, detail=f"No face embeddings found for faceInfoId: {face_id}")

    embeddings = {}
    
    for direction, embedding_blob in faces:
        try:
            if embedding_blob:
                embedding = pickle.loads(embedding_blob)
                if hasattr(embedding, 'tolist'):
                    embeddings[direction] = embedding.tolist()
                else:
                    embeddings[direction] = embedding
        except Exception as e:
            print(f"Error deserializing embedding for direction {direction}: {str(e)}")
            continue

    if not embeddings:
        raise HTTPException(status_code=404, detail="Could not deserialize any embeddings")

    return {
        "faceInfoId": face_id,
        "embeddings": embeddings
    }

async def get_face_embedding_by_direction(face_id: int, direction: str, db: AsyncSession):
    result = await db.execute(
        select(Face.embedding)
        .where(Face.faceInfoId == face_id)
        .where(Face.direction == direction)
    )
    embedding_blob = result.scalar()

    if not embedding_blob:
        raise HTTPException(
            status_code=404, 
            detail=f"No face embedding found for faceInfoId: {face_id} and direction: {direction}"
        )

    try:
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        embedding_data = embedding.tolist()
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error deserializing embedding: {str(e)}"
        )

    return {
        "faceInfoId": face_id,
        "direction": direction,
        "embedding": embedding_data
    }