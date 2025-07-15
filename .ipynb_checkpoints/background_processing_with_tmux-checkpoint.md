# Running and Detaching a `tmux` Session for Background Processes

To create a `tmux` session named 'klue', run a process within it, and then detach it to keep it running in the background, use the following commands:

## 1. Create and Start a New `tmux` Session

```bash
tmux new -s klue
```
Use `-s` for session-name

This command creates a new tmux session named 'klue' and immediately enters it. Once inside, you can execute any commands you want to run.

For example, 
```bash
$ ./run full
```

## 2. Detach from the Session
After starting your task inside the session, if you want to keep it running in the background, press the following key combination:

```bash
Ctrl+b d
```
(Hold down the Ctrl key, press b, then release both keys and press d.)

This action detaches you from the currently active 'klue' session, returning you to your terminal. Any programs running inside the 'klue' session will continue to execute in the background.

```bash
(klue) ~$ tmux new -s klue
[detached (from session klue)]
(klue) ~$
```

## 3. Reattach to a detached tmux session
After a while, you come back to the terminal and forgot the name of the session.

### List running tmux sessions
```Bash
tmux ls
  #or
tmux list-sessions
```
Using these commands will show your 'klue' session listed as 'detached'.

### Reattach to the detached session
```Bash
tmux attach -t klue
```
Use `-t` for target-session.

This command allows you to re-enter the 'klue' session you previously detached from, letting you view output or regain control of the running programs.