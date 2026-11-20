def _row_log_policy(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def _root_rotation_policy(x, sid):
    """Average two presentations of the same own-seat board before root action selection."""
    worker, base = _POLICY.forward(x, sid)
    rotated_worker, rotated_base = _POLICY.forward(rotate_features(x), sid)
    # Rows follow cell order. Worker directions turn with the board, recruitment labels do not.
    restored_worker = rotated_worker[::-1, :][:, (0, 3, 4, 1, 2)]
    restored_base = rotated_base[::-1, :]
    worker_mixture = np.logaddexp(_row_log_policy(worker), _row_log_policy(restored_worker))
    base_mixture = np.logaddexp(_row_log_policy(base), _row_log_policy(restored_base))
    log_two = np.float32(np.log(2.0))
    return worker_mixture - log_two, base_mixture - log_two
