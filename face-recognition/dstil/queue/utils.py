import json
import base64


def unwrap_message(properties, body):
    """
    Unwraps the given message body by inspecting the content-type
    and decoding the body accordingly.
    :param properties: BasicProperties object with message metadata
    :param body: Body of the message
    :return: JSON representation of the message
    """
    # Ensure content-type is supported.
    content_type = get_content_type(properties)
    if content_type != "application/json":
        raise ValueError("Invalid content-type: " + content_type)

    # Decode message into a JSON object.
    obj = json.loads(body.decode("utf8"))
    if obj["contentType"] == "application/json":
        json_bytes = base64.decodebytes(obj["bytes"].encode("utf8"))
        json_str = json_bytes.decode("utf8")
        obj["json"] = json.loads(json_str)

    return obj


def get_content_type(properties):
    """
    Returns the content-type of the message as set by Spring Cloud Stream in the headers.
    :param properties: RabbitMQ BasicProperties object
    :return: The message content-type as a string
    """
    if "originalContentType" in properties.headers:
        original_content_type = properties.headers["originalContentType"]
        if ";" in original_content_type:
            return original_content_type.split(";")[0]
        else:
            return original_content_type
    elif "contentType" in properties.headers:
        return properties.headers["contentType"]
    else:
        return properties.content_type