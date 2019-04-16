from flask import Flask
from .config import Config

app = Flask(__name__)
app.config.from_object(Config)


from model import output

def runoverwrite(main=None, argv=None):
    import sys as _sys
    from tensorflow.python.platform import flags
    f = flags.FLAGS
    args = argv[1:] if argv else None
    flags_passthrough = f._parse_flags(args=args)
    main(_sys.argv[:1] + flags_passthrough)



textObject = output.Text_recognition()
runoverwrite(textObject.load_model)

from app import routes


