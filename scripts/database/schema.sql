CREATE TABLE IF NOT EXISTS "games" (
    "id" serial NOT NULL,
    "bgg_id" integer NOT NULL,
    "name" varchar(255) NOT NULL,
    "description" varchar(65535),
    "created_at" timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY ("id")
);

CREATE TABLE IF NOT EXISTS "comments" (
    "id" serial NOT NULL,
    "user" varchar(255) NOT NULL,
    "rating" real NOT NULL,
    "comment" text NOT NULL,
    "comment_lang" text,
    "game_id" integer NOT NULL,
    "created_at" timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY ("id")
);