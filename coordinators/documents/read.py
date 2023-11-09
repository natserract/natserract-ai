from database import connect

async def count_documents():
    try:
        conn = await connect()
        results = await conn.fetchval(
            f"select count(*) from documents")

        return results
    except Exception as e:
        raise ValueError("Can't get documents")


async def get_all():
    try:
        conn = await connect()
        results = await conn.fetch(
            f"select * from documents docs")

        return results
    except Exception as e:
        raise ValueError('Tenant not found!')