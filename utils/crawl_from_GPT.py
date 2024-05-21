import openai
import time
from config import *

key_count = [0] * (len(key_pool))


def get_response_from_message(messages, _model_name, max_token=None, wait=10, timeout=60):
    count = 0
    index = key_count.index(min(key_count))
    while True:
        try:
            openai.api_key = key_pool[index]
            if max_token is not None:
                resp = openai.ChatCompletion.create(model=_model_name, timeout=timeout,
                                                    messages=messages,
                                                    max_tokens=max_token)
            else:
                resp = openai.ChatCompletion.create(model=_model_name, timeout=timeout,
                                                    messages=messages, )
            key_count[index] += 1
            resp = resp['choices'][0]['message']['content']
            if resp is None:
                print('wrong sample!')
            return resp

        except Exception as e:
            count += 1
            if "billing" in str(e):
                key_count[index] = 99999999
                print(f'API token {index} out of money, switch')
                index = key_count.index(min(key_count))
            elif "Rate limit" in str(e):
                print(f'API token {index} out of rate, switch')
                index = key_count.index(min(key_count))
                continue
            elif "timeout" in str(e):
                print(f'Timeout!')
                index = key_count.index(min(key_count))
                print(messages)
                continue
            # elif "4097" in str(e):
            #     print('Exceed max token limit, reducing 100')
            #     if max_token is not None:
            #         max_token = max_token - 100
            #         if max_token <= 0:
            #             return None
            #     index = key_count.index(min(key_count))
            else:
                print(e)
                print("Warning: no scripts to handle this kind of error, please write one to avoid infinite looping!!!")
            time.sleep(wait)
            if count == 100:
                print('exceed maximum retry, skip the message and return None')
                return None
