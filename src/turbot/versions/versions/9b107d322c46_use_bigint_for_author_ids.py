"""Use BIGINT for author ids

Revision ID: 9b107d322c46
Revises: 1afdca2a2389
Create Date: 2020-05-17 11:34:03.356515

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "9b107d322c46"
down_revision = "1afdca2a2389"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("bugs") as b:
        b.alter_column("author", existing_type=sa.Integer(), type_=sa.BigInteger())

    with op.batch_alter_table("fossils") as b:
        b.alter_column("author", existing_type=sa.Integer(), type_=sa.BigInteger())

    with op.batch_alter_table("songs") as b:
        b.alter_column("author", existing_type=sa.Integer(), type_=sa.BigInteger())

    with op.batch_alter_table("art") as b:
        b.alter_column("author", existing_type=sa.Integer(), type_=sa.BigInteger())

    with op.batch_alter_table("fish") as b:
        b.alter_column("author", existing_type=sa.Integer(), type_=sa.BigInteger())

    with op.batch_alter_table("prices") as b:
        b.alter_column("author", existing_type=sa.Integer(), type_=sa.BigInteger())

    with op.batch_alter_table("users") as b:
        b.alter_column("author", existing_type=sa.Integer(), type_=sa.BigInteger())


def downgrade():
    with op.batch_alter_table("bugs") as b:
        b.alter_column("author", existing_type=sa.BigInteger(), type_=sa.Integer())

    with op.batch_alter_table("fossils") as b:
        b.alter_column("author", existing_type=sa.BigInteger(), type_=sa.Integer())

    with op.batch_alter_table("songs") as b:
        b.alter_column("author", existing_type=sa.BigInteger(), type_=sa.Integer())

    with op.batch_alter_table("art") as b:
        b.alter_column("author", existing_type=sa.BigInteger(), type_=sa.Integer())

    with op.batch_alter_table("fish") as b:
        b.alter_column("author", existing_type=sa.BigInteger(), type_=sa.Integer())

    with op.batch_alter_table("prices") as b:
        b.alter_column("author", existing_type=sa.BigInteger(), type_=sa.Integer())

    with op.batch_alter_table("users") as b:
        b.alter_column("author", existing_type=sa.BigInteger(), type_=sa.Integer())
