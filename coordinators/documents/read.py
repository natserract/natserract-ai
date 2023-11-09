from database import connect


async def get_all():
    try:
        conn = await connect()
        results = await conn.fetch(
            f"select * from documents docs")

        return results
    except Exception as e:
        raise ValueError('Tenant not found!')