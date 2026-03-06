"""Quick smoke-test for the API endpoints."""
import urllib.request
import json
import uuid


BASE = "http://127.0.0.1:8000"


def test_upload():
    boundary = uuid.uuid4().hex
    with open("test_data.csv", "rb") as f:
        file_data = f.read()

    body = (
        "--" + boundary + "\r\n"
        'Content-Disposition: form-data; name="file"; filename="test_data.csv"\r\n'
        "Content-Type: text/csv\r\n"
        "\r\n"
    ).encode() + file_data + ("\r\n--" + boundary + "--\r\n").encode()

    req = urllib.request.Request(
        BASE + "/api/dataset/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    print("=== UPLOAD ===")
    print(json.dumps(result, indent=2))
    return result["id"]


def test_analysis(dataset_id):
    req = urllib.request.Request(BASE + f"/api/dataset/{dataset_id}/analysis")
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    print("\n=== ANALYSIS ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    did = test_upload()
    test_analysis(did)
    print("\nAll tests passed!")
