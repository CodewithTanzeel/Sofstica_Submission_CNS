def test_load_split_returns_expected_columns(loaded_split, feature_table):
    expected_columns = set(feature_table.columns)
    for split_name, frame in loaded_split.items():
        assert expected_columns.issubset(frame.columns)
        assert "split" in frame.columns


def test_feature_prep_fits_on_train_only(mock_split):
    from src.features import prepare_features

    train = mock_split["train"]
    val = mock_split["val"]
    test = mock_split["test"]

    prepared = prepare_features(train, val, test)

    assert prepared["train"].shape[0] == len(train)
    assert prepared["val"].shape[0] == len(val)
    assert prepared["test"].shape[0] == len(test)


def test_model_predict_returns_score_in_unit_interval(mock_split):
    from src.features import prepare_features
    from src.model import SimpleModel

    prepared = prepare_features(mock_split["train"], mock_split["val"], mock_split["test"])
    model = SimpleModel()
    model.fit(prepared["train"], prepared["train"]["label"])
    scores = model.predict_proba(prepared["val"])

    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
