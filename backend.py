from confluent_kafka import Consumer, KafkaException, KafkaError
import json
from nar_algorithm2npy import al2npy

# Kafka 서버 주소와 토픽 설정
KAFKA_SERVER = 'nar.kafka:9093'  # Kafka 서버 주소
TOPIC = 'queuedevel'  # 수신할 토픽

# KafkaConsumer 설정
consumer = Consumer({
    'bootstrap.servers': KAFKA_SERVER,
    'group.id': 'prediction',
    'auto.offset.reset': 'earliest'  # 가장 첫 번째 메시지부터 읽기
})

def main():
    print(f"Kafka Consumer - Waiting for messages from topic '{TOPIC}'...")

    # Kafka 토픽에 구독
    consumer.subscribe([TOPIC])

    try:
        while True:
            # 메시지 수신
            msg = consumer.poll(timeout=1.0)  # 1초 동안 대기
            
            if msg is None:
                # 메시지가 없으면 계속 대기
                continue
            elif msg.error():
                # 에러가 발생한 경우
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    print(f"End of partition reached {msg.partition()} at offset {msg.offset()}")
                else:
                    raise KafkaException(msg.error())
            else:
                # 정상적인 메시지가 수신되면 출력
                data = json.loads(msg.value().decode('utf-8')) 
                print(data)

                uuid = data['uuid']
                seq_name = data['seq_name']
                sequence = data['sequence']
                algorithm = data['algorithm']
                
                # print(uuid, seq_name, sequence, algorithm)
                
                al2npy(algorithm=algorithm, uuid=uuid, seq_name=seq_name, sequence=sequence, base_path='/data')
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # 종료 시 Consumer 닫기
        consumer.close()

if __name__ == '__main__':
    main()
