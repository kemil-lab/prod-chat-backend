from app.db.Chroma_clientV2 import get_collection


def getResult():
    print(1)

    collection = get_collection()

    if collection.count() == 0:
        return {"message": "No embedded items found in collection"}

    result = collection.query(
        query_texts=["What is MS"],
        n_results=3,
    )

    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    formatted = []

    for i in range(len(documents)):
        formatted.append({
            "id": ids[i] if i < len(ids) else None,
            "content": documents[i],
            "metadata": metadatas[i] if i < len(metadatas) else {},
            "distance": distances[i] if i < len(distances) else None,
        })

    return formatted


if __name__ == "__main__":
    print(getResult())