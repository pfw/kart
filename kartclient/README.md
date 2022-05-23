# kartclient 

Will invoke commands via a unix socket to `kart helper`. The helper will be started if not already running. 
By default it uses a socket at `$HOME/.kart.socket` and the helper will shutdown after `300s` if not commands were 
received.

`kart helper` will set up a forked child with an environment that matches the environment `kartclient` was 
called from, e.g. `stdin/stdout/stderr`, current directory, environment variables and arguments passed.

## To setup for build:
```cmake -S . -B build```

## To build:
```cmake --build build```

## To run (eg):
`kartclient` is called with all of the same command and options as `kart`, those commands are passed to the helper 
and the result returned.

```./build/kartclient -C ~/Documents/Koordinates/California-National-Parks status```