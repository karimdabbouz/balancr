from .extensions import scheduler
from .utils import compute_lda_topics
from flask import current_app as app


@scheduler.task('interval', seconds=60*300)
def compute_lda_topics_task():
    with scheduler.app.app_context():
        compute_lda_topics()