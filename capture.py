import cv2
import requests
import numpy as np

folder = "./data/"
if __name__ == '__main__':
    # prefix = input("Name of the sherd: ")
    prefix='cal'
    # define convex side as up(1) side, concave side as down(0) side
    up_prefix = prefix
    down_prefix = prefix+"-0-"
    up_count = 0
    down_count = 0
    upside_time = 1
    downside_time = 0
    r = requests.get('http://10.111.137.36:8080?action=stream', stream=True)
    if (r.status_code == 200):
        bytes = bytes()
        for chunk in r.iter_content(chunk_size=1024):
            bytes += chunk
            a = bytes.find(b'\xff\xd8')
            b = bytes.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes[a:b + 2]
                bytes = bytes[b + 2:]
                i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.imshow('i', i)

                if cv2.waitKey(1) == 32:
                    filename = ""
                    if up_count < upside_time:
                        filename = up_prefix + str(up_count) + ".jpg"
                        cv2.imwrite(folder + filename, i)
                        print(filename + " saved.")
                        up_count += 1
                    elif down_count < downside_time:
                        filename = down_prefix + str(down_count) + ".jpg"
                        cv2.imwrite(folder + filename, i)
                        print(filename + " saved.")
                        down_count += 1

                # elif cv2.waitKey(1) == 27:
                #     exit(0)
    else:
        print("Received unexpected status code {}".format(r.status_code))