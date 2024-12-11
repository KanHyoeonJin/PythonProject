import threading
import cv2
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import numpy as np
from collections import deque
import sys #Python의 stdout이 버퍼링 때문에 버퍼를 강제로 비워주는 것
# YOLO 모델 로드
model = YOLO("yolov8n.pt")
LOGGER.setLevel("ERROR")
# 카메라가 시작되어야 청소가 되게 하는 플래그
camera_ready = False
yolo_ready = False
room_clean = [False, False, False, False]

# 객체 감지 결과를 저장할 전역 변수
detected_objects = []
lock = threading.Lock()

def camera_thread():
    global detected_objects,camera_ready,yolo_ready
    print("카메라 스레드 시작")

    cap = cv2.VideoCapture(0)  # 카메라 연결
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    camera_ready = True
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        frame_count +=1
        # YOLO 감지 수행
        if frame_count % 2 == 0:

            results = model(frame)  # YOLO 모델 호출
            objects = []

            for result in results:  # 결과 리스트 처리
                for box in result.boxes:  # Bounding box 정보
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # 좌표 가져오기
                    confidence = box.conf[0].item()  # 신뢰도
                    class_id = int(box.cls[0].item())  # 클래스 ID
                    label = model.names[class_id]  # 클래스 이름
                    if label in ["person", "dog", "cat"]:
                        objects.append((label, (x1, y1, x2, y2)))
            if not yolo_ready:
                yolo_ready = True

        # 객체 감지 결과를 안전하게 업데이트
            with lock:
                detected_objects = objects

        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키로 종료
            break

    cap.release()
    cv2.destroyAllWindows()


def cleaning_simulation():
    global detected_objects,camera_ready,yolo_ready
    print("청소 스레드 시작")
    while not (camera_ready and yolo_ready):
        print("카메라 기다리기")
        cv2.waitKey(500)
    if camera_ready and yolo_ready :
         print("카메라 준비 완료")
         main()
        # 객체 감지 결과 확인  
    
    
   
    while True:

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

free_space_map = cv2.imread("map.png", cv2.IMREAD_GRAYSCALE)

# 이미지가 제대로 로드되었는지 확인
if free_space_map is None:
    print("이미지 파일을 읽을 수 없습니다. 경로를 확인하세요.")
    exit()

# 방을 탐지하기 위해 라벨링 (방 구분)
def label_rooms(free_space_map):
    binary_map = (free_space_map == 255).astype(np.uint8)
    num_labels, labeled_map = cv2.connectedComponents(binary_map, connectivity=4)

    # 방 2와 방 3의 라벨 값을 임시 저장 후 교환
    room_2_label = (labeled_map == 3).astype(np.uint8)
    room_3_label = (labeled_map == 4).astype(np.uint8)

    labeled_map[room_2_label == 1] = 10  # 임시 라벨로 방 2 변경
    labeled_map[room_3_label == 1] = 3   # 방 3으로 교환
    labeled_map[labeled_map == 10] = 4   # 방 2로 교환

    return labeled_map, num_labels


# 라벨링된 맵 생성
labeled_map, num_rooms = label_rooms(free_space_map)

# 방문 여부를 관리하는 2D 배열
visited_positions = np.zeros_like(free_space_map, dtype=bool)  # False로 초기화

# 현재 방의 모든 좌표를 반환
def get_room_coordinates(labeled_map, current_room):
    room_coordinates = set(zip(*np.where(labeled_map == current_room)))
    filtered_coordinates = {(y,x) for y,x in room_coordinates if free_space_map[y,x] ==255}
    print(f"Room {current_room}: Total coordinates={len(room_coordinates)}, Filtered={len(filtered_coordinates)}")
    return filtered_coordinates

# 가장 가까운 흰색 영역까지 경로를 계산하는 함수 (BFS)
def find_path_to_white(start_position, free_space_map, labeled_map, current_room, allow_gray=False):
    start_position = tuple(start_position)  # 시작 위치를 튜플로 변환

    # 현재 위치가 이미 흰색 영역이고 방문되지 않았다면 바로 반환
    if (
        free_space_map[start_position[0], start_position[1]] == 255
        and not visited_positions[start_position[0], start_position[1]]
    ):
        return [start_position]

    queue = deque([start_position])
    visited = set()
    parent = {start_position: None}  # 시작 위치의 부모는 None으로 설정

    while queue:
        current_y, current_x = queue.popleft()

        if (current_y, current_x) in visited:
            continue
        visited.add((current_y, current_x))

        # 흰색 영역을 찾으면 경로를 추적
        if (
            0 <= current_y < free_space_map.shape[0]
            and 0 <= current_x < free_space_map.shape[1]
            and free_space_map[current_y, current_x] == 255
            and not visited_positions[current_y, current_x]  # 아직 청소되지 않은 영역
            and labeled_map[current_y, current_x] == current_room
        ):
            path = []
            while parent[(current_y, current_x)] is not None:
                path.append((current_y, current_x))
                current_y, current_x = parent[(current_y, current_x)]
            path.reverse()
            return path
        # 상하좌우 이동 (검은색은 제외, 회색도 제외할 수 있음)
        for dy, dx in [(-4, 0), (4, 0), (0, -4), (0, 4)]:
            ny, nx = current_y + dy, current_x + dx
            if (
                0 <= ny < free_space_map.shape[0]
                and 0 <= nx < free_space_map.shape[1]
                and (ny, nx) not in visited
                and (ny, nx) not in parent  # 부모가 없는 새로운 위치만 추가
                and free_space_map[ny, nx] != 0  # 검은색 영역(벽)은 제외
                and (allow_gray or free_space_map[ny, nx] != 200)  # 회색 영역도 지나가지 않도록
            ):
                queue.append((ny, nx))
                parent[(ny, nx)] = (current_y, current_x)

    return []  # 경로를 찾지 못한 경우 빈 리스트 반환

# 로봇의 위치를 그리는 함수
def draw_robot(map_with_robot, position, free_space_map):
    for y in range(map_with_robot.shape[0]):
        for x in range(map_with_robot.shape[1]):
            if free_space_map[y, x] == 0:  # 검은색(벽) 유지
                map_with_robot[y, x] = [0, 0, 0]
            elif 100 <= free_space_map[y, x] <= 200:  # 회색(출입구) 유지
                map_with_robot[y, x] = [free_space_map[y, x]] * 3

    # 흰색 영역에만 노란색으로 칠하기
    if free_space_map[position[0], position[1]] == 255:
        cv2.circle(map_with_robot, (position[1], position[0]), 4, (0, 255, 255), -1)  # 노란색으로 칠함

    temp_map = map_with_robot.copy()
    cv2.circle(temp_map, (position[1], position[0]), 4, (0, 0, 255), -1)  # 레드닷
    return temp_map
# 정지 함수
def update_display(map_with_robot, robot_position, free_space_map):
    map_display = draw_robot(map_with_robot, robot_position, free_space_map)
    cv2.imshow("Robot Movement Simulation", map_display)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()

# 청소하는 함수들

def clean_room(robot_position, labeled_map, free_space_map, map_with_robot, room_number):
    global room_clean
    if (room_number - 1 == 0):
        print("청소 중인 구역 : 거실")
    else:
        print(f"청소 중인 구역 : 방 {room_number - 1}")

    path_to_target = find_path_to_white(robot_position, free_space_map, labeled_map, room_number, allow_gray=True)
    
    if path_to_target:
        for step in path_to_target:
            with lock:
                objects = detected_objects.copy()

            # 사람 감지 시 동작
            if any(label == 'person' for label, bbox in objects):
                print("사람 감지: 로봇이 잠시 멈춥니다.")
                cv2.waitKey(2000)  # 로봇 멈춤

                # 사람 인식 시 다음 이동 경로에 검은색 칠하기
                if free_space_map[step[0], step[1]] == 255:  # 다음 경로가 흰색 영역인 경우
                    print("사람 인식: 다음 이동 경로에 검은색 칠하기.")
                    y, x = step
                    free_space_map[y - 3:y + 3, x - 3:x + 3] = 0  # 6x6 검은색 칠하기
                    map_with_robot[y - 3:y + 3, x - 3:x + 3] = [0, 0, 0]  # 컬러 맵 업데이트

                    # 검은색 칠한 후 화면 갱신
                    update_display(map_with_robot, robot_position, free_space_map)
                    cv2.waitKey(1000)  # 검은색 칠한 후 잠시 멈춤

                    # 검은색 칠한 후 새로운 경로 계산
                    print("새 경로를 계산합니다.")
                    path_to_target = find_path_to_white(robot_position, free_space_map, labeled_map, room_number, allow_gray=True)
                    if not path_to_target:
                        print("새로운 경로를 찾을 수 없습니다. 종료합니다.")
                        return robot_position
                    break  # 현재 루프를 중단하고 새로운 경로로 이동

            # 로봇이 검은색을 칠한 후에 다시 이동
            robot_position = step
            visited_positions[step[0],step[1]] = True
            update_display(map_with_robot, robot_position, free_space_map)

    room_coordinates = get_room_coordinates(labeled_map, room_number)

    total_cleanable = len(room_coordinates)
     # 청소되지 않은 방의 좌표를 찾고 계속 청소
   

    while not all(visited_positions[y, x] for y, x in room_coordinates):
        path_to_white = find_path_to_white(robot_position, free_space_map, labeled_map, room_number, allow_gray=False)
        if path_to_white:
            for step in path_to_white:
                with lock:
                    objects = detected_objects.copy()

                # 사람 감지 시 동작
                if any(label == 'person' for label, bbox in objects):
                    print("사람 감지: 로봇이 잠시 멈춥니다.")
                    cv2.waitKey(2000)

                    # 사람 인식 시 다음 이동 경로에 검은색 칠하기
                    if free_space_map[step[0], step[1]] == 255:  # 다음 경로가 흰색 영역인 경우
                        print("사람 인식: 다음 이동 경로에 검은색 칠하기.")
                        y, x = step
                        free_space_map[y - 3:y + 3, x - 3:x + 3] = 0  # 6x6 검은색 칠하기
                        map_with_robot[y - 3:y + 3, x - 3:x + 3] = [0, 0, 0]

                        # 검은색 칠한 후 화면 갱신
                        update_display(map_with_robot, robot_position, free_space_map)
                        cv2.waitKey(1000)  # 검은색 칠한 후 잠시 멈춤

                        # 검은색 칠한 후 새로운 경로 계산
                        print("새 경로를 계산합니다.")
                        path_to_white = find_path_to_white(robot_position, free_space_map, labeled_map, room_number, allow_gray=True)
                        if not path_to_white:
                            print("새로운 경로를 찾을 수 없습니다. 종료합니다.")
                            return robot_position
                        break  # 현재 루프를 중단하고 새로운 경로로 이동

                # 로봇이 검은색을 칠한 후에 다시 이동
                robot_position = step
                if not visited_positions[step[0], step[1]]:
                    visited_positions[step[0],step[1]]=True
                
                map_with_robot[step[0], step[1]] = [0, 255, 255]  # 청소된 영역 표시
                update_display(map_with_robot, robot_position, free_space_map)
        else:
            break
    cleaned_positions = sum(visited_positions[y,x] for y,x in room_coordinates)*16
   
    print(f"청소된 좌표 수 : {cleaned_positions} / {total_cleanable}")
    cleaned_percentage = (cleaned_positions/total_cleanable) * 100
    print(f"Room {room_number - 1}: Cleaned={cleaned_positions}, Total={total_cleanable}, Percentage={cleaned_percentage:.2f}%")

    if cleaned_percentage>=70:
        room_clean[room_number - 1] = True
        print(f"{'거실' if room_number - 1 == 0 else f'방 {room_number - 1}'} 청소 완료")
    else:
        print(f"{'거실' if room_number - 1 == 0 else f'방 {room_number - 1}'} 청소 실패. 청소되지 않은 영역이 남아 있습니다.")
       


    return robot_position



#초기위치로 돌아가는 함수
def find_path_to_position(start_position, target_position, free_space_map):
    start_position = tuple(start_position)
    target_position = tuple(target_position)

    if start_position == target_position:
        return [start_position]  # 이미 도착한 경우

    queue = deque([start_position])
    visited = set()
    parent = {start_position: None}  # 경로 추적을 위한 부모 딕셔너리

    while queue:
        current_y, current_x = queue.popleft()

        if (current_y, current_x) in visited:
            continue
        visited.add((current_y, current_x))

        # 목표 위치에 도달하면 경로를 추적
        if (current_y, current_x) == target_position:
            path = []
            while parent[(current_y, current_x)] is not None:
                path.append((current_y, current_x))
                current_y, current_x = parent[(current_y, current_x)]
            path.reverse()
            return path

        # 상하좌우로 이동 (검은색 영역 제외)
        for dy, dx in [(-4, 0), (4, 0), (0, -4), (0, 4)]:
            ny, nx = current_y + dy, current_x + dx
            if (
                0 <= ny < free_space_map.shape[0]
                and 0 <= nx < free_space_map.shape[1]
                and (ny, nx) not in visited
                and (ny, nx) not in parent
                and free_space_map[ny, nx] != 0  # 검은색 영역 제외
            ):
                queue.append((ny, nx))
                parent[(ny, nx)] = (current_y, current_x)

    print("경로를 찾지 못했습니다.")
    return []  # 경로를 찾지 못한 경우 빈 리스트 반환


# 메인 시뮬레이션 함수
def main():
    robot_position = (200, 420)  # 로봇 초기 위치 설정
    map_with_robot = cv2.cvtColor(free_space_map, cv2.COLOR_GRAY2BGR)  # 맵 복사 (컬러 변환)
    robot_position = clean_room(robot_position, labeled_map, free_space_map, map_with_robot, 1)  # 거실 청소
    robot_position = clean_room(robot_position, labeled_map, free_space_map, map_with_robot, 2)  # 방 1 청소
    robot_position = clean_room(robot_position, labeled_map, free_space_map, map_with_robot, 3)  # 방 2 청소
    robot_position = clean_room(robot_position, labeled_map, free_space_map, map_with_robot, 4)  # 방 3 청소
    if all(room_clean):
        print("모든 청소가 완료되었습니다.")
    else:
        print("청소가 완료되지 않은 방이 있습니다.")

    # 마지막 위치에서 (200, 420)으로 돌아가는 경로 계산
    return_to_start_path = find_path_to_position(robot_position, (200, 420), free_space_map)

    # 경로를 따라 로봇 이동
    if return_to_start_path:
        for step in return_to_start_path:
            robot_position = step
            update_display(map_with_robot, robot_position, free_space_map)

    

def visualize_labeled_map(labeled_map):
    # 라벨 맵을 컬러 이미지로 변환
    labeled_map_color = np.zeros((labeled_map.shape[0], labeled_map.shape[1], 3), dtype=np.uint8)

    # 고유 라벨 확인
    unique_labels = np.unique(labeled_map)

    # 라벨별 경계선 그리기
    for label in unique_labels:
        if label == 0:  # 배경은 제외
            continue
        
        # 특정 라벨의 마스크 생성
        mask = (labeled_map == label).astype(np.uint8) * 255
        
        # 경계선 검출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 방 내부를 랜덤 색상으로 칠하기
        color = tuple(np.random.randint(0, 255, size=3).tolist())  # 랜덤 색상
        labeled_map_color[labeled_map == label] = color
        
        # 경계를 초록색으로 표시
        cv2.drawContours(labeled_map_color, contours, -1, (0, 255, 0), 2)

    # 시각화
    cv2.imshow("Labeled Map with Boundaries", labeled_map_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
     # 방 라벨링
    labeled_map, num_rooms = label_rooms(free_space_map)
    
    # 라벨링 맵 시각화
    visualize_labeled_map(labeled_map)

    # 두 작업을 병렬로 실행
    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    clean_thread = threading.Thread(target=cleaning_simulation, daemon=True)

    # 스레드 시작
    cam_thread.start()
    print("카메라스레드")
    clean_thread.start()
    print("청소스레드")

    # 종료 대기
    cam_thread.join()
    clean_thread.join()
