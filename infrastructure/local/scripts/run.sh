set -e

cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1
DIR="$PWD"

DOCKER_CONTAINER_NAME="destark"

if [ "$(docker ps -aq -f name=$DOCKER_CONTAINER_NAME)" ]; then
    # cleanup
    docker container rm $DOCKER_CONTAINER_NAME
fi

# Rebuild image unless SKIP_REBUILD=true
[ "$SKIP_REBUILD" = "true" ] || docker  build --file Dockerfile --network=host -t destaque-territorial-spark . --platform linux/amd64

# run in a new container
docker run -it \
    --env-file $DIR/.env \
    --mount type=bind,source="${DIR}",target=/destark \
    --name "$DOCKER_CONTAINER_NAME" \
    --network=host \
    destaque-territorial-spark \
    "$@"

# remove container
docker container rm $DOCKER_CONTAINER_NAME > /dev/null
