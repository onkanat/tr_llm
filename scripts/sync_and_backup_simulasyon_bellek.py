#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
simulasyon_bellek Yedekleme ve Senkronizasyon Betiği
===================================================
1. localhost:6333 üzerindeki 4.073 adet belgeyi tüm vektör ve metadata'larıyla çeker.
2. 'data/simulasyon_bellek_export.jsonl' dosyasına kalıcı yedek olarak kaydeder.
3. Yerel dosya tabanlı 'data/qdrant_db' deposuna 'simulasyon_bellek' koleksiyonunu yazar.
Böylece Docker/sunucu kapalıyken bile tüm sistem offline çalışabilir hale gelir.
"""

import os
import sys
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 70)
    print(" SIMULASYON_BELLEK YEDEKLEME VE SENKRONİZASYON")
    print("=" * 70)

    remote_host = "127.0.0.1"
    remote_port = 6333
    collection_name = "simulasyon_bellek"
    export_path = "data/simulasyon_bellek_export.jsonl"
    local_db_path = "data/qdrant_db"

    # 1. Connect to remote
    print(f"\n[1] Uzak Qdrant sunucusuna ({remote_host}:{remote_port}) bağlanılıyor...")
    try:
        remote_client = QdrantClient(host=remote_host, port=remote_port, timeout=5.0)
        total_remote = remote_client.count(collection_name).count
        print(f"  ✓ Bağlantı başarılı. '{collection_name}' koleksiyonunda {total_remote} kayıt bulundu.")
    except Exception as e:
        print(f"Hata: Uzak Qdrant sunucusuna bağlanılamadı: {e}")
        return

    # 2. Scroll and export all points with vectors
    print(f"\n[2] Tüm noktalar çekiliyor ve '{export_path}' dosyasına aktarılıyor...")
    batch_size = 250
    offset = None
    all_points = []
    
    with open(export_path, "w", encoding="utf-8") as f_out:
        while True:
            records, offset = remote_client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            if not records:
                break
            
            for r in records:
                vecs = r.vector
                vec_dict = {}
                if isinstance(vecs, dict):
                    if "dense" in vecs:
                        vec_dict["dense"] = list(vecs["dense"])
                    if "sparse" in vecs:
                        sp = vecs["sparse"]
                        if hasattr(sp, "indices"):
                            vec_dict["sparse"] = {"indices": list(sp.indices), "values": list(sp.values)}
                        elif isinstance(sp, dict):
                            vec_dict["sparse"] = sp
                point_data = {
                    "id": r.id,
                    "payload": r.payload,
                    "vectors": vec_dict
                }
                all_points.append(point_data)
                f_out.write(json.dumps(point_data, ensure_ascii=False) + "\n")
                
            print(f"  -> {len(all_points)} / {total_remote} nokta dışa aktarıldı...")
            if offset is None:
                break

    print(f"  ✓ Başarılı: Toplam {len(all_points)} nokta '{export_path}' dosyasına yedeklendi.")

    # 3. Synchronize to local data/qdrant_db
    print(f"\n[3] Yerel dosya deposuna ('{local_db_path}') senkronize ediliyor...")
    local_client = QdrantClient(path=local_db_path)
    
    # Check if collection exists in local
    if local_client.collection_exists(collection_name):
        local_count = local_client.count(collection_name).count
        print(f"  Yerel depoda mevcut '{collection_name}' koleksiyonu bulundu ({local_count} kayıt). Yeniden oluşturuluyor...")
        local_client.delete_collection(collection_name)
        
    local_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=768, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }
    )

    # Upsert points in batches
    batch_upload = 100
    for i in range(0, len(all_points), batch_upload):
        chunk = all_points[i:i + batch_upload]
        points_to_upsert = []
        for p in chunk:
            dense_vec = p["vectors"].get("dense")
            sparse_raw = p["vectors"].get("sparse")
            
            sparse_vec = models.SparseVector(
                indices=sparse_raw.get("indices", []),
                values=sparse_raw.get("values", [])
            ) if isinstance(sparse_raw, dict) else None
            
            vector_dict = {"dense": dense_vec}
            if sparse_vec:
                vector_dict["sparse"] = sparse_vec
                
            points_to_upsert.append(models.PointStruct(
                id=p["id"],
                vector=vector_dict,
                payload=p["payload"]
            ))
            
        local_client.upsert(collection_name=collection_name, points=points_to_upsert)
        print(f"  ✓ Yerel depoya aktarıldı: {min(i + batch_upload, len(all_points))} / {len(all_points)}")

    final_local_count = local_client.count(collection_name).count
    print(f"\n[TAMAMLANDI] Yerel '{local_db_path}' deposundaki '{collection_name}' sayısı: {final_local_count}")

if __name__ == "__main__":
    main()
