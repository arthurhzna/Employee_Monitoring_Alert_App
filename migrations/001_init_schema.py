from yoyo import step

steps = [

step("""
CREATE TABLE device (
    device_id SERIAL PRIMARY KEY,
    device_name VARCHAR(100) UNIQUE NOT NULL,
    is_regist BOOLEAN DEFAULT FALSE
);
"""),

step("""
CREATE TABLE person (
    person_id SERIAL PRIMARY KEY,
    uuid VARCHAR(100) UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    device_id INT REFERENCES device(device_id)
);
"""),

step("""
CREATE TABLE face_recog (
    face_recog_id SERIAL PRIMARY KEY,
    predict VARCHAR(100),
    person_id INT REFERENCES person(person_id)
);
"""),

step("""
CREATE TABLE behavior (
    behavior_id SERIAL PRIMARY KEY,
    predict VARCHAR(100),
    person_id INT REFERENCES person(person_id)
);
"""),

step("""
CREATE TABLE drowsiness (
    drowsiness_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    person_id INT REFERENCES person(person_id)
);
"""),

step("""
CREATE TABLE bbox (
    bbox_id SERIAL PRIMARY KEY,
    width INT,
    height INT,
    person_id INT REFERENCES person(person_id)
);
"""),

step("""
CREATE TABLE dwelling_time (
    dwelling_id SERIAL PRIMARY KEY,
    dwelling_looking INT,
    dwelling_not_looking INT,
    person_id INT REFERENCES person(person_id)
);
"""),

step("""
CREATE INDEX idx_person_device
ON person(device_id);
"""),

step("""
CREATE INDEX idx_person_timestamp
ON person(timestamp);
"""),

step("""
CREATE INDEX idx_bbox_person
ON bbox(person_id);
"""),

step("""
CREATE INDEX idx_behavior_person
ON behavior(person_id);
"""),

step("""
CREATE INDEX idx_face_recog_person
ON face_recog(person_id);
"""),

step("""
CREATE INDEX idx_drowsiness_person
ON drowsiness(person_id);
"""),

step("""
CREATE INDEX idx_dwelling_person
ON dwelling_time(person_id);
""")

]