# CUDA version
CUDA 9.1

# python (conda)
conda create -n env_name python=3.6

# pytorch
pip install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

# tenserflow
pip install tensorflow-gpu==1.14.0

# 사용법
nar_algorithm2npy.py 실행하면 되는데, 마지막 al2npy 을 보면 됨.\
numpy 형식의 n*n 매트릭스를 반환 (write_ct_file 형식으로 ct 파일 생성하는 함수는 짜두고 사용하진 않았으니 적당히 보고 사용하면 됨)

# frontend 에서 보낼 양식과, 받을 양식
```
Request
path: /register_uuid
type: POST
header: {
  uuid: string // 작업의 uuid
}
body: {
  seq_name: string, // 시퀀스 이름
  sequence: string, // A, U, G, C 로 이루어진 시퀀스
  algorithm: string, // spot, e2e, red
  iteration: int, // SimRNA 반복 횟수
}

서버 동작
- uuid 에 해당하는 폴더 혹은 db 를 등록함
- 등록이 완료되었다는 Response 를 보냄
- 프로세스 시작

Response
body: {
  data: {
    msg: string, // 메시지
    success: boolean, // 동록 성공 여부 (이미 등록된 uuid 라도 여기선 등록 성공 메시지를 보냄)
  }
}

어플리케이션 동작
- 성공 메시지가 날아오면 uuid 를 쿼리스트링에 실어 결과페이지로 이동
```

```
Request
path: /get_data
type: GET
query_string: {
  u: string, // 작업의 uuid
}

서버 동작
- 들어온 uuid 를 조회하여 결과 반환
- uuid 가 없는 uuid 일 경우 -> 에러 메시지 반환
- 아직 작업 중인 uuid 의 경우 -> 진행과정 (%) 반환
- 작업이 끝나서 결과가 저장된 uuid 의 경우 -> 결과 반환

Response
body: {
  state: int, // 0: 없는 uuid, 1: 작업 중인 uuid, 2: 작업이 끝난 uuid
  msg: string, // 메시지
  progress: int, // state == 0, 2: -1, state == 1: 작업 경과 퍼센트
  seq_name: string, // 시퀀스 이름
  sequence: string, // 시퀀스
  dot_bracket: string, // ex. '...((([.)))...((.]))'
  image_2d: string, // png 이미지의 base64
  pdb_3d: string, // pdb 데이터의 base64
}

어플리케이션 동장
- 받은 결과를 바탕으로 결과 페이지 생성
```