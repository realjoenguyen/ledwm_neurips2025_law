from ledwm.embodied.core.logger import AsyncOutput
import numpy as np
from termcolor import cprint
import concurrent.futures
from ledwm import jaxutils
try:
    from ledwm.startup import configure_tensorflow_cpp_warnings
except ModuleNotFoundError:
    from startup import configure_tensorflow_cpp_warnings


def _encode_gif(frames, fps):
    from subprocess import Popen, PIPE

    h, w, c = frames[0].shape
    pxfmt = {1: "gray", 3: "rgb24"}[c]
    cmd = " ".join(
        [
            "ffmpeg -y -f rawvideo -vcodec rawvideo",
            f"-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex",
            "[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse",
            f"-r {fps:.02f} -f gif -",
        ]
    )
    proc = Popen(cmd.split(" "), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tobytes())
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError("\n".join([" ".join(cmd), err.decode("utf8")]))
    del proc
    return out


class TensorBoardOutput(AsyncOutput):
    def __init__(self, logdir, fps=5, maxsize=1e9, parallel=True, reset=False):
        super().__init__(self._write, parallel)
        self._logdir = str(logdir)
        cprint(f'Logging to TensorBoard at "{self._logdir}".')
        # if reset:
        #     self._delete_old_event_files()

        if self._logdir.startswith("/gcs/"):
            self._logdir = self._logdir.replace("/gcs/", "gs://")
        self._fps = fps
        self._writer = None
        self._maxsize = self._logdir.startswith("gs://") and maxsize
        if self._maxsize:
            self._checker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._promise = None

    # def _delete_old_event_files(self):
    #     event_files = tf.io.gfile.glob(
    #         self._logdir.rstrip("/") + "/events.out.tfevents.*"
    #     )
    #     for file in event_files:
    #         try:
    #             tf.io.gfile.remove(file)
    #             cprint(f"Deleted old TensorBoard file: {file}", "red")
    #         except Exception as e:
    #             print(f"Error deleting file {file}: {e}")

    def _write(self, summaries):
        configure_tensorflow_cpp_warnings()
        import tensorflow as tf

        reset = False
        if self._maxsize:
            result = self._promise and self._promise.result()
            # print('Current TensorBoard event file size:', result)
            reset = self._promise and result >= self._maxsize
            self._promise = self._checker.submit(self._check)

        if not self._writer or reset:
            cprint(
                f"Creating new TensorBoard event file writer, at {self._logdir}.",
                "green",
            )

            self._writer = tf.summary.create_file_writer(
                self._logdir, flush_millis=1000, max_queue=10000
            )
        self._writer.set_as_default()
        for step, name, value in summaries:
            if "text" in name:
                continue
            try:
                if isinstance(value, str):
                    tf.summary.text(name, value, step)

                elif len(value.shape) == 0:
                    tf.summary.scalar(name, value, step)

                elif len(value.shape) == 1:
                    if len(value) > 1024:
                        value = value.copy()
                        np.random.shuffle(value)
                        value = value[:1024]
                    tf.summary.histogram(name, value, step)

                elif len(value.shape) == 2:
                    tf.summary.image(name, value, step)

                elif len(value.shape) == 3:
                    tf.summary.image(name, value, step)

                elif len(value.shape) == 4 and (
                    value.shape[-1] == 1 or value.shape[-1] == 3
                ):
                    self._video_summary(name, value, step)

                elif len(value.shape) == 4 and "openl" in name:
                    # image = MessengerSent.make_image(value)
                    print(f"processing {name} with shape {value.shape}")
                    if "model" in name:
                        value = jaxutils.video_from_image_model(value)
                    self._video_summary(name, value, step)
                else:
                    cprint(
                        f"logger.tensorboard_skip | name={name} | shape={value.shape}",
                        "red",
                    )

            except Exception:
                print(f"logger.tensorboard_write_error | name={name}")
                raise

        self._writer.flush()
        print(f"logger.tensorboard_flush | directory={self._logdir}")

    def _check(self):
        configure_tensorflow_cpp_warnings()
        import tensorflow as tf

        events = tf.io.gfile.glob(self._logdir.rstrip("/") + "/events.out.*")
        return tf.io.gfile.stat(sorted(events)[-1]).length if events else 0

    def _video_summary(self, name, video, step):
        configure_tensorflow_cpp_warnings()
        import tensorflow as tf
        from subprocess import Popen, PIPE
        import tensorflow.compat.v1 as tf1

        name = name if isinstance(name, str) else name.decode("utf-8")
        if np.issubdtype(video.dtype, np.floating):
            video = np.clip(255 * video, 0, 255).astype(np.uint8)
        try:
            T, H, W, C = video.shape
            summary = tf1.Summary()
            image = tf1.Summary.Image(height=H, width=W, colorspace=C)
            image.encoded_image_string = _encode_gif(video, self._fps)
            summary.value.add(tag=name, image=image)
            tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)

        except (IOError, OSError) as e:
            print(f"logger.gif_error | reason=ffmpeg_unavailable | error={e}")
            tf.summary.image(name, video, step)
