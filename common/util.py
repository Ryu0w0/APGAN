def notify(message):
    import requests
    url = "https://notify-api.line.me/api/notify"
    token = ""
    headers = {"Authorization": "Bearer " + token}

    message = message
    payload = {"message": message}

    requests.post(url, headers=headers, params=payload)
