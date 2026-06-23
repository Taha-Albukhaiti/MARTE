PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS credentials;
DROP TABLE IF EXISTS users;

CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    salary INTEGER NOT NULL
);

CREATE TABLE credentials (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    password_hash TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

INSERT INTO users (id, name, role, salary) VALUES
    (1, 'Alice',   'admin',    120000),
    (2, 'Bob',     'manager',   90000),
    (3, 'Charlie', 'employee',  65000),
    (4, 'Diana',   'employee',  62000),
    (5, 'Eve',     'employee',  70000);

INSERT INTO credentials (id, user_id, password_hash) VALUES
    (1, 1, 'hash_alice'),
    (2, 2, 'hash_bob'),
    (3, 3, 'hash_charlie'),
    (4, 4, 'hash_diana'),
    (5, 5, 'hash_eve');
