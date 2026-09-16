"""Container liveness check; does not submit tasks or require GPU availability."""
import json
import urllib.request


def main():
    # Ignore inherited HTTP proxies for the local control service.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:8002/api/health', timeout=4) as response:
        data = json.load(response)['data']
        if data.get('status') != 'ok' or data.get('mode') != 'single-host':
            raise RuntimeError('unexpected platform health response')


if __name__ == '__main__':
    main()
