import mysql.connector
import logging

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',  # MySQL username
    'password': '00125680',  # MySQL password
    'database': 'shehjad_vehicle_data'  # MySQL database name
}

def insert_data(video_timestamp,qTime, cross_type, track_id, confidence, vehicle_class, image_path):
    try:
        connection = mysql.connector.connect(**db_config)
        print("Connection to MySQL database successful")

        # Create a cursor object
        cursor = connection.cursor()
        print(type(video_timestamp))
        qTime_str = str(qTime)
        #cross_type = str(cross_type)

        print(f"Type of video_timestamp: {type(video_timestamp)}")
        print(f"Type of qTime: {type(qTime)}")
        print(f"Type of cross_type: {type(cross_type)}")
        print(f"Type of track_id: {type(track_id)}")
        print(f"Type of confidence: {type(confidence)}")
        print(f"Type of vehicle_class: {type(vehicle_class)}")
        print(f"Type of image_path: {type(image_path)}")

        # SQL query to insert data into the table
        insert_query = "INSERT INTO location (video_timestamp, qTime, cross_type, track_id, confidence, vehicle_class, img_path )VALUES (%s,%s,%s,%s,%s,%s,%s)"

        # Make sure to pass a tuple for the parameters
        cursor.execute(insert_query, (video_timestamp, qTime_str, cross_type, track_id, confidence, vehicle_class, image_path))

        # Commit the changes
        connection.commit()
        print(f"Data for video_timestamp {video_timestamp} and qTime {qTime_str} inserted successfully")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        # Closing the cursor and connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        print("Connection closed")

def insert_video_name(video_name, complete_flag):
    try:
        # Connect to the database
        connection = mysql.connector.connect(**db_config)
        print("Connection to MySQL database successful")

        # Create a cursor object
        cursor = connection.cursor()

        # SQL query to insert data into the video table
        insert_query = "INSERT INTO info (name, complete) VALUES (%s, %s)"

        # Execute the query with the parameters
        cursor.execute(insert_query, (video_name, complete_flag))

        # Commit the changes to the database
        connection.commit()
        print(f"Data for video_name '{video_name}' with complete_flag '{complete_flag}' inserted successfully")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        # Closing the cursor and connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        print("Connection closed")


def update_complete_flag(video_name: str) -> None:
    """
    Update the complete_flag in the database for a video.
    """
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        update_query = "UPDATE info SET complete= %s WHERE name = %s"
        cursor.execute(update_query, (True, video_name))
        connection.commit()
        print(f"Updated complete_flag for video '{video_name}'")
    except mysql.connector.Error as err:
        print(f"Error updating complete_flag for video '{video_name}': {err}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def insert_analytics_file_path(file_path: str) -> None:
    """
    Insert the analytics file path into the database.
    
    Args:
        file_path (str): The path where the analytics file is stored.
    """
    try:
        # Establish a database connection
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # SQL query to insert only the file path
        insert_query = """
        INSERT INTO analytics_file (path)
        VALUES (%s)
        """

        # Execute the query with the provided file path
        cursor.execute(insert_query, (file_path,))
        connection.commit()
        
        logging.info(f"Analytics file path '{file_path}' inserted successfully into the database.")
    
    except mysql.connector.Error as err:
        logging.error(f"Error inserting analytics file path '{file_path}': {err}")
    
    finally:
        # Close cursor and connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def show_all_config_files() -> None:
    """
    Retrieve and print all configuration file paths from the database.
    """
    try:
        # Establish a database connection
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # SQL query to retrieve all file paths
        select_query = "SELECT id, path FROM analytics_file"

        # Execute the query
        cursor.execute(select_query)

        # Fetch all results
        results = cursor.fetchall()

        if results:
            print("Available configuration files:")
            for row in results:
                print(f"{row[0]} - {row[1]}")  # Print each file path
        else:
            print("No configuration files found in the database.")

    except mysql.connector.Error as err:
        logging.error(f"Error retrieving configuration files: {err}")
    
    finally:
        # Close cursor and connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def get_config_path_by_id(config_id: int) -> str:
    """
    Retrieve the configuration file path for a specific ID from the database.
    
    Args:
        config_id (int): The ID of the configuration file.
        
    Returns:
        str: The configuration file path if found, else None.
    """
    try:
        # Establish a database connection
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # SQL query to retrieve the file path for the specified ID
        select_query = "SELECT path FROM analytics_file WHERE id = %s"

        # Execute the query with the given config_id
        cursor.execute(select_query, (config_id,))

        # Fetch the result
        result = cursor.fetchone()

        # Return the path if found, otherwise return None
        return result[0] if result else None

    except mysql.connector.Error as err:
        logging.error(f"Error retrieving configuration file for ID {config_id}: {err}")
        return None
    
    finally:
        # Close cursor and connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()