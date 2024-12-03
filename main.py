import asyncio
import struct
import aioble
import bluetooth
import time
from collections import OrderedDict
import json
from TinyS3Led import TinyS3Led 

# UUID del servizio IMU e delle caratteristiche
IMU_SERVICE_UUID = bluetooth.UUID("0000ffe5-0000-1000-8000-00805f9a34fb")
IMU_CHAR_READ_UUID = bluetooth.UUID("0000ffe4-0000-1000-8000-00805f9a34fb")
IMU_CHAR_WRITE_UUID = bluetooth.UUID("0000ffe9-0000-1000-8000-00805f9a34fb")

class CustomOrderDict(OrderedDict):
    def __str__(self):
        return "{" + ", ".join(f"{k}: {v}" for k, v in self.items()) + "}"

# Dizionario dei risultati
results = CustomOrderDict()
results["Timestamp"] = 0

results["AccX"] = 0
results["AccY"] = 0
results["AccZ"] = 0

results["GyroX"] = 0
results["GyroY"] = 0
results["GyroZ"] = 0

results["AngX"] = 0
results["AngY"] = 0
results["AngZ"] = 0

results["MagX"] = 0
results["MagY"] = 0
results["MagZ"] = 0

# Variabile per il primo timestamp
initial_timestamp = None

#Variabile contatore
counter = 0

#Variabile per il led
led = TinyS3Led()

def append_to_json_file(filename, entry):
    global counter

    # Scrive i dati aggiornati nel file
    with open(filename, 'a') as f:
        if counter == 0:
            f.write("[")
        json.dump(entry, f)
        f.write(",\n")
    
    counter += 1

# Conversione di byte in numero intero
def _get_signed_int(data):
    return struct.unpack("<h", bytes(data))[0]

# Trova il sensore tramite MAC
async def find_sensor():
    while True:
        async with aioble.scan(5000, interval_us=30000, window_us=30000, active=True) as scanner:
            async for result in scanner:
                mac = ":".join("{:02X}".format(b) for b in result.device.addr)
                if mac == "C5:6B:CD:C8:57:A4" and IMU_SERVICE_UUID in result.services():
                    return result
    return None

# genera nomi per file json univoci e sequenziali
def get_unique_filename(base_name="imu_data"):
    i = 1
    while True:
        filename = f"{base_name}_{i}.json"
        try:
            with open(filename, "r"):
                pass
        except OSError:
            return filename
        i += 1

# Decodifica i dati dall'IMU
async def _decode_data(imu_char_read):
    data = await imu_char_read.notified()
    
    if len(data) == 20:
        Ax = _get_signed_int(data[2:4]) / 32768 * 16
        Ay = _get_signed_int(data[4:6]) / 32768 * 16
        Az = _get_signed_int(data[6:8]) / 32768 * 16
        Gx = _get_signed_int(data[8:10]) / 32768 * 2000
        Gy = _get_signed_int(data[10:12]) / 32768 * 2000
        Gz = _get_signed_int(data[12:14]) / 32768 * 2000
        AngX = _get_signed_int(data[14:16]) / 32768 * 180
        AngY = _get_signed_int(data[16:18]) / 32768 * 180
        AngZ = _get_signed_int(data[18:20]) / 32768 * 180

        results["AccX"] = Ax
        results["AccY"] = Ay
        results["AccZ"] = Az
        
        results["GyroX"] = Gx
        results["GyroY"] = Gy
        results["GyroZ"] = Gz
        
        results["AngX"] = AngX
        results["AngY"] = AngY
        results["AngZ"] = AngZ
        

# Riceve i dati del magnetometro
async def process_magnetometer_data(imu_char_read):
    magnetometer_data = await imu_char_read.notified()
    
    if len(magnetometer_data) >= 10: 
        Hx = _get_signed_int(magnetometer_data[4:6]) / 120  
        Hy = _get_signed_int(magnetometer_data[6:8]) / 120
        Hz = _get_signed_int(magnetometer_data[8:10]) / 120
        
        results["MagX"] = Hx
        results["MagY"] = Hy
        results["MagZ"] = Hz

# Calcola il nuovo timestamp in millisecondi
def timer():
    global initial_timestamp
    
    if initial_timestamp is None:
        initial_timestamp = time.time_ns() // 1_000_000

    current_timestamp = time.time_ns() // 1_000_000
    relative_timestamp = current_timestamp - initial_timestamp
    results["Timestamp"] = relative_timestamp
    

async def main():
    led.red()
    
    file_name = get_unique_filename("imu_data")
    device = await find_sensor()

    if device is None:
        print("IMU sensor not found\n")
        return

    try:
        connection = await device.device.connect()
    except asyncio.TimeoutError:
        print("Timeout during connection")
        return


    async with connection:
        led.green()
        
        try:
            imu_service = await connection.service(IMU_SERVICE_UUID)
            imu_char_read = await imu_service.characteristic(IMU_CHAR_READ_UUID)
            imu_char_write = await imu_service.characteristic(IMU_CHAR_WRITE_UUID)

            await imu_char_read.subscribe(notify=True)
            
            # ne legge in media 4 al secondo
            while True:
                led.blue()
                
                timer()
                await _decode_data(imu_char_read)
                await imu_char_write.write(bytes([0xFF, 0xAA, 0x27, 0x3A, 0x00]))
                await process_magnetometer_data(imu_char_read)
                
                append_to_json_file(file_name, results)
                
                print(results)

                await asyncio.sleep(0.1)

        except asyncio.TimeoutError:
            print("Timeout discovering services/characteristics")
            led.red()
            
        except aioble.DeviceDisconnectedError:
            print("Device Disconnected")
            led.red()
                       
# Esegui la funzione principale asincrona
asyncio.run(main())
