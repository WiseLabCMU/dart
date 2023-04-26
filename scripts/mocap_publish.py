# OptiTrack NatNet direct depacketization sample for Python 3.x
#
# Uses the Python NatNetClient.py library to establish a connection (by creating a NatNetClient),
# and receive data via a NatNet connection and decode it using the NatNetClient library.
# Receives data that is intercepted by udp_relay.c, and republishes rigid bodies to an MQTT Topic. 
# Uses the Paho Python and MQTT libraries. 


import paho.mqtt.client as mqtt
import json
from datetime import datetime as dt

from NatNetClient import NatNetClient


#The broker and port to connect to send MQTT messages.
BROKER_URL = 'mqtt.arenaxr.org'
BROKER_PORT = 8883
BROKER_TOPIC = 'realm/s/simple/'

RIGID_BODY_ID = 11


mqtt_client = None
last_position = None
last_rotation = None


# Callback at end of frame. Sends rigid body pose with frame timestamp
def receiveFrame(frameNumber, markerSetCount, unlabeledMarkersCount,
                 rigidBodyCount, skeletonCount, labeledMarkerCount,
                 timecode, timecodeSub, timestamp, 
                 isRecording, trackedModelsChanged):
    global mqtt_client
    global last_position
    global last_rotation
    # print('got frame')
    if mqtt_client is not None and \
        last_position is not None and \
            last_rotation is not None:
        message = {
            "object_id": "mmwave-radar",
            "data": {
                "position": {
                    "x": last_position[0],
                    "y": last_position[1],
                    "z": last_position[2]
                },
                "rotation": {
                    "x": last_rotation[0],
                    "y": last_rotation[1],
                    "z": last_rotation[2],
                    "w": last_rotation[3]
                },
                "timestamp": timestamp
            }
        }
        mqtt_client.publish(BROKER_TOPIC, json.dumps(message), qos=1, retain=False)
        last_position = None
        last_rotation = None
        # print('sent pose')


# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
def receiveRigidBodyFrame(id, position, rotation):
    global last_position
    global last_rotation
    # print('got rigid body')
    if id == RIGID_BODY_ID:
        last_position = position
        last_rotation = rotation
        # print('got right id')
       

def on_connect(client, userdata, flags, rc):
    print(f'Connected With Result Code {rc}')


def main():
    global mqtt_client

    # This will create a new NatNet client
    mocap_client = NatNetClient()

    # Connect to the MQTT client
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.tls_set()
    mqtt_client.username_pw_set(username="cli",password="eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImFyZW5heHIub3JnIn0.eyJzdWIiOiJjbGkiLCJhdWQiOiJhcmVuYSIsInJvb20iOiIqIiwic3VicyI6WyIjIl0sInB1YmwiOlsiIyJdLCJpc3MiOiJhcmVuYS1hY2NvdW50IiwiaWF0IjoxNjQzMTQ0MDE3LCJleHAiOjE2NzQ2ODAwMTd9.UZuhiwn65Bb237R9zjU1QO0up8W2AKpsYFscfhp-qSZ--X95gBdRkwnr-qxQFPyG690wGlvpOTaRbIlw49_hSw5NIznjiOowKbjnGS6A7FKurgIxIyiCIapRzgkNrFdw3I9NhVnWn3OKQD5FUJ8fxHdUu483xz7Vj97DnzCp1VP2RITbIVvbom7uQFDwOLdup_Uo4WpwhU1D0Uleo2wKpdS0X_tsMaswiSRY4pK2bWZEx7dkdq3hCjzk1NYen7B5XN9m-I6i4_2BaNM5F5zFP23GKZN63S3UUw3st7XBchWssl0PM1TLIKg6bM_TvhSvvJVQHsF_X5E9uWw-TryRpW-yGcwhDxs82tjqL1uF8e7Lg_nQjSK7ykURLVxUKtvyk4mKR5SGsRruyZeSqNbZLH96hl0IX32Tze2TQh2oonkkkdKn57Ku4sAu4CWET94Od6AoVchdhXJ1HOLFb3nSbcq4Ykk_0k-Q-gbaPiOvdxrgjNyh06F6C6tXi4aR8cLMAXI2wIZQ3EnXPu6zepabOxhB4MLlJdHnMrBq_l2tGRZgqA7oSBcsS5dRc0EmCyJT4cr02CVF7ReKqoaA-JpZt-2KWCAHRIF9tyEmnDdacVtHIl-jyBsSvyPznL4C6Kw5naL8wNI1eeEtpiH1y2AVHHtdRvVb67UqT5u1gYiZ-_Y")
    mqtt_client.connect(BROKER_URL, BROKER_PORT)

    # Configure the streaming client to call our rigid body handler on the emulator to send data out.
    mocap_client.newFrameListener = receiveFrame
    mocap_client.rigidBodyListener = receiveRigidBodyFrame

    # Start up the streaming client now that the callbacks are set up.
    # This will run perpetually, and operate on a separate thread.
    mocap_client.run()

    # Loop the mqtt client so it doesn't disconnect
    mqtt_client.loop_forever()


if __name__ == '__main__':
    main()
