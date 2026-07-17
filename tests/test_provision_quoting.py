"""Tests for vast_provision shell-command construction.

Specifically guards against the filename-with-spaces bug that bit twice:
1. The launch command — "Paul Deposition.prepped.wav" got split into 2 args,
   python received "Paul" as audio path, errored.
2. The polling command — the unquoted [ -f {done_flag} ] test split the path
   and falsely returned RUNNING forever.
"""
import vast_provision as vp


class TestBuildLaunchCommand:
    def test_simple_filename(self):
        cmd = vp.build_launch_command(
            audio_name="audio.wav",
            base_stem="audio",
            hf_token="hf_dummy",
        )
        # Simple names need no quoting; shlex.quote leaves them alone
        assert "canary_transcribe.py audio.wav" in cmd

    def test_filename_with_space_is_quoted(self):
        # The thing that broke Paul. The audio_name "Paul Deposition.prepped.wav"
        # MUST end up wrapped in single quotes in the resulting shell command.
        cmd = vp.build_launch_command(
            audio_name="Paul Deposition.prepped.wav",
            base_stem="Paul Deposition.prepped",
            hf_token="hf_dummy",
        )
        # The audio path should appear with quoting so it's one arg
        assert "'Paul Deposition.prepped.wav'" in cmd
        # And NOT appear bare
        assert "canary_transcribe.py Paul Deposition.prepped.wav" not in cmd

    def test_done_flag_with_space_is_quoted(self):
        cmd = vp.build_launch_command(
            audio_name="Paul Deposition.prepped.wav",
            base_stem="Paul Deposition.prepped",
            hf_token="hf_dummy",
        )
        # The done.flag path with space must be quoted in the rm
        assert "'Paul Deposition.prepped.done.flag'" in cmd

    def test_hf_token_inline(self):
        # HF_TOKEN must be set inline before the python invocation
        # (SSH sessions don't inherit Vast's --env)
        cmd = vp.build_launch_command(
            audio_name="audio.wav", base_stem="audio", hf_token="hf_secret"
        )
        assert "HF_TOKEN=hf_secret" in cmd

    def test_uses_nohup_and_redirect(self):
        # Detached: nohup + stdin from /dev/null + stdout/stderr to run.log
        cmd = vp.build_launch_command(
            audio_name="audio.wav", base_stem="audio", hf_token="hf_dummy"
        )
        assert "nohup" in cmd
        assert "< /dev/null" in cmd
        assert "> run.log 2>&1" in cmd

    def test_runs_in_background(self):
        # The launch must end in & (within a subshell) so SSH closes
        cmd = vp.build_launch_command(
            audio_name="audio.wav", base_stem="audio", hf_token="hf_dummy"
        )
        # The subshell-background pattern: ( ... &)
        assert "&)" in cmd


class TestBuildPollCommand:
    def test_simple_base_stem(self):
        cmd = vp.build_poll_command(base_stem="audio")
        # Simple stems need no quoting
        assert "/root/work/audio.done.flag" in cmd
        assert "/root/work/audio.$n.progress.log" in cmd

    def test_base_stem_with_space_quoted_in_done_flag_test(self):
        # The thing that hung Paul polling. The [ -f ... ] test must receive
        # one quoted argument, not multiple unquoted words.
        cmd = vp.build_poll_command(base_stem="Paul Deposition.prepped")
        # The done.flag path with space must be quoted
        assert "'/root/work/Paul Deposition.prepped.done.flag'" in cmd
        # And NOT appear bare in a [ -f ... ] test
        assert "[ -f /root/work/Paul Deposition.prepped.done.flag ]" not in cmd

    def test_base_stem_with_space_quoted_in_progress_loop(self):
        # The for-loop's progress.log path also has the space, also needs quoting
        cmd = vp.build_poll_command(base_stem="Paul Deposition.prepped")
        # The base_stem appears in the loop's f=... assignment, which needs quotes
        assert "'Paul Deposition.prepped'" in cmd

    def test_returns_one_of_done_running_dead(self):
        # The command should echo one of these three statuses
        cmd = vp.build_poll_command(base_stem="audio")
        assert "echo DONE" in cmd
        assert "echo RUNNING" in cmd
        assert "echo DEAD" in cmd

    def test_progress_marker_present(self):
        # The --progress-- delimiter the provisioner parses
        cmd = vp.build_poll_command(base_stem="audio")
        assert "---progress---" in cmd
