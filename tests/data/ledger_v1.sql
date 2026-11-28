-- A v1.0 ledger (user_version 0), as the code before #41 creates and fills it.
BEGIN TRANSACTION;
CREATE TABLE usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    candidates_tokens INTEGER NOT NULL,
    thoughts_tokens INTEGER NOT NULL,
    cost_gbp REAL NOT NULL,
    cleared INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE reservations (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    amount_gbp REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE TABLE session_caps (
    session_id TEXT PRIMARY KEY,
    cap_gbp REAL NOT NULL
);
CREATE TABLE entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry TEXT NOT NULL
);
INSERT INTO usage VALUES(1,'legacy-sess',100,50,10,0.005,0);
INSERT INTO usage VALUES(2,'legacy-sess',80,20,5,0.002,1);
INSERT INTO usage VALUES(3,'demo:x',40,10,0,0.001,0);
INSERT INTO reservations VALUES('res-legacy','legacy-sess',0.0015,4102444800.0);
INSERT INTO session_caps VALUES('legacy-sess',0.03);
INSERT INTO entries VALUES(1,'{"time":"2026-01-02T03:04:05Z","source":"web","session_id":"legacy-sess","label":"screen","outcome":"ALLOW","cost_gbp":0.005}');
INSERT INTO entries VALUES(2,'{"time":"2026-01-02T03:04:06Z","source":"web","session_id":"demo:x","label":"screen","outcome":"ALLOW","cost_gbp":0.001}');
CREATE INDEX usage_session ON usage (session_id, cleared);
COMMIT;
