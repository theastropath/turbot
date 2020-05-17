"""Adds relationships

Revision ID: 1afdca2a2389
Revises: cb8aa87d74cb
Create Date: 2020-05-15 15:13:06.433119

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "1afdca2a2389"
down_revision = "cb8aa87d74cb"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("prices") as b:
        b.create_foreign_key("fk_prices_author", "users", ["author"], ["author"])

    with op.batch_alter_table("fossils") as b:
        b.create_foreign_key("fk_fossils_author", "users", ["author"], ["author"])

    with op.batch_alter_table("art") as b:
        b.create_foreign_key("fk_art_author", "users", ["author"], ["author"])

    with op.batch_alter_table("fish") as b:
        b.create_foreign_key("fk_fish_author", "users", ["author"], ["author"])

    with op.batch_alter_table("bugs") as b:
        b.create_foreign_key("fk_bugs_author", "users", ["author"], ["author"])

    with op.batch_alter_table("songs") as b:
        b.create_foreign_key("fk_songs_author", "users", ["author"], ["author"])


def downgrade():
    with op.batch_alter_table("prices") as b:
        b.drop_constraint("fk_prices_author", type_="foreignkey")

    with op.batch_alter_table("fossils") as b:
        b.drop_constraint("fk_fossils_author", type_="foreignkey")

    with op.batch_alter_table("art") as b:
        b.drop_constraint("fk_art_author", type_="foreignkey")

    with op.batch_alter_table("fish") as b:
        b.drop_constraint("fk_fish_author", type_="foreignkey")

    with op.batch_alter_table("bugs") as b:
        b.drop_constraint("fk_bugs_author", type_="foreignkey")

    with op.batch_alter_table("songs") as b:
        b.drop_constraint("fk_songs_author", type_="foreignkey")
