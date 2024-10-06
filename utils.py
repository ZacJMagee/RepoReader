import re

class UtilityFunctions:
    @staticmethod
    def sanitize_collection_name(name):
        """
        Sanitize the collection name to ensure it meets the required format.
        """
        # Remove any character that's not alphanumeric, underscore, or hyphen
        name = re.sub(r'[^\w\-]', '_', name)
        
        # Ensure the name starts and ends with an alphanumeric character
        name = re.sub(r'^[^\w]+', '', name)
        name = re.sub(r'[^\w]+$', '', name)
        
        # Replace consecutive underscores with a single underscore
        name = re.sub(r'_{2,}', '_', name)
        
        # Ensure the name is between 3 and 63 characters
        if len(name) < 3:
            name = name.ljust(3, 'a')
        if len(name) > 63:
            name = name[:63]
        
        # Ensure the name is not a valid IPv4 address
        if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', name):
            name = 'collection_' + name
        
        return name

    @staticmethod
    def ensure_alternating_roles(conversation_history):
        """
        Ensure that the conversation history has alternating roles.
        """
        processed_messages = []
        last_role = None
        for message in conversation_history:
            if message["role"] != last_role:
                processed_messages.append(message)
                last_role = message["role"]
        
        # Ensure the conversation starts with a user message
        if processed_messages and processed_messages[0]["role"] == "assistant":
            processed_messages.pop(0)
        
        return processed_messages
